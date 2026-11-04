"""Business API: keys, full partner flow, error mapping and certificate checks."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from temoin import service
from temoin.api import api, get_settings
from temoin.config import Settings
from temoin.schemas import ChallengeCheck, Comparison, Report

KEY = "partner-key-123"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(180, 170, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def api_settings(tmp_path: Path, test_settings: Settings) -> Settings:
    """Settings with an isolated data folder and one partner key."""
    return dataclasses.replace(test_settings, data_dir=tmp_path / "data", api_keys=(KEY,))


@pytest.fixture
def client(
    api_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    sale_report: Report,
    comparison: Comparison,
) -> TestClient:
    """An API client whose model steps are faked (no network)."""

    def fake_report(photo: Path, expected_code: str, **_: Any) -> Report:
        challenge = ChallengeCheck(
            expected_code=expected_code, visible=True, read_code=expected_code
        )
        return sale_report.model_copy(update={"challenge": challenge})

    monkeypatch.setattr(service, "create_report", fake_report)
    monkeypatch.setattr(service, "run_investigation", lambda *_, **__: comparison)
    api.dependency_overrides[get_settings] = lambda: api_settings
    yield TestClient(api)
    api.dependency_overrides.clear()


def _auth() -> dict[str, str]:
    return {"X-API-Key": KEY}


def test_health_needs_no_key(client: TestClient) -> None:
    """The liveness probe is public."""
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_calls_without_valid_key_are_rejected(client: TestClient) -> None:
    """Missing or wrong keys get 401."""
    assert client.post("/v1/cases").status_code == 401
    assert client.post("/v1/cases", headers={"X-API-Key": "nope"}).status_code == 401


def test_api_is_closed_without_configured_keys(api_settings: Settings) -> None:
    """No TEMOIN_API_KEYS means the API refuses every call (secure by default)."""
    api.dependency_overrides[get_settings] = lambda: dataclasses.replace(api_settings, api_keys=())
    try:
        response = TestClient(api).post("/v1/cases", headers=_auth())
    finally:
        api.dependency_overrides.clear()
    assert response.status_code == 503


def test_full_partner_flow_and_verification(client: TestClient, comparison: Comparison) -> None:
    """Open, sale photo, buyer code, delivery photo, status, then verify the certificate."""
    opened = client.post("/v1/cases", headers=_auth())
    assert opened.status_code == 201
    case_id, sale_code = opened.json()["case_id"], opened.json()["code"]
    assert sale_code in opened.json()["instructions"]

    sale = client.post(
        f"/v1/cases/{case_id}/sale",
        headers=_auth(),
        files={"photo": ("sale.jpg", _jpeg(), "image/jpeg")},
        data={"declared_materials": "cuir, métal"},
    )
    assert sale.status_code == 200, sale.text
    assert sale.json()["fresh"] is True

    code = client.post(f"/v1/cases/{case_id}/delivery-code", headers=_auth())
    assert code.status_code == 201
    assert code.json()["code"] != sale_code

    delivery = client.post(
        f"/v1/cases/{case_id}/delivery",
        headers=_auth(),
        files={"photo": ("delivery.jpg", _jpeg(), "image/jpeg")},
    )
    assert delivery.status_code == 200, delivery.text
    body = delivery.json()
    assert body["verdict"] == comparison.verdict
    assert body["confidence"] == comparison.confidence

    status = client.get(f"/v1/cases/{case_id}", headers=_auth()).json()
    assert status["has_sale_constat"] and status["has_delivery_constat"]
    assert status["verdicts"] == [comparison.verdict]

    certificate = body["certificate"]
    checked = client.post("/v1/verify", headers=_auth(), json=certificate).json()
    assert checked["intact"] is True and checked["chained"] is False
    certificate["comparison"]["confidence"] = 100 - certificate["comparison"]["confidence"]
    assert client.post("/v1/verify", headers=_auth(), json=certificate).json()["intact"] is False


def test_unknown_case_is_404(client: TestClient) -> None:
    """A case id that does not exist is reported as not found."""
    assert client.get("/v1/cases/tx-00000000", headers=_auth()).status_code == 404


def test_delivery_before_sale_is_a_conflict(client: TestClient) -> None:
    """Steps out of order are refused with a clear 409."""
    case_id = client.post("/v1/cases", headers=_auth()).json()["case_id"]
    assert client.post(f"/v1/cases/{case_id}/delivery-code", headers=_auth()).status_code == 409


def test_non_image_upload_is_rejected(client: TestClient) -> None:
    """Only photo formats are accepted."""
    case_id = client.post("/v1/cases", headers=_auth()).json()["case_id"]
    response = client.post(
        f"/v1/cases/{case_id}/sale",
        headers=_auth(),
        files={"photo": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_docs_use_relative_urls_behind_proxies(client: TestClient) -> None:
    """Docs load the schema relatively and 'Try it out' targets the mount path."""
    page = client.get("/docs")
    assert page.status_code == 200
    assert "url: 'openapi.json'" in page.text
    assert client.get("/openapi.json").json()["servers"] == [{"url": "."}]


def test_unexpected_errors_answer_json_with_the_error_type(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash inside a route returns JSON naming the error type, not a bare 500."""

    def boom() -> Settings:
        raise RuntimeError("secret detail")

    api.dependency_overrides[get_settings] = boom
    try:
        response = TestClient(api, raise_server_exceptions=False).get(
            "/v1/cases/tx-00000000", headers={"X-API-Key": KEY}
        )
    finally:
        api.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json() == {"detail": "Erreur interne de Témoin.", "error": "RuntimeError"}
