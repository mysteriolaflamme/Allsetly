"""Tests for monocular AI depth (offline, fake network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from temoin.__main__ import main
from temoin.config import Settings
from temoin.monodepth import MonoDepthError, estimate_depth, model_path
from temoin.stereo import DepthMatrix


class _FakeNet:
    """Standalone network stub returning a fixed depth map."""

    def __init__(self, depth: np.ndarray) -> None:
        """Store the fixed depth map."""
        self._depth = depth

    def setInput(self, blob: object) -> None:
        """Ignore the input blob."""

    def forward(self) -> np.ndarray:
        """Return the fixed depth map."""
        return self._depth


def test_estimate_depth_normalizes_fake_output(test_settings: Settings) -> None:
    """A ramp map becomes ordered 0..1 mono relief."""
    ramp = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    matrix = estimate_depth(Image.new("RGB", (8, 8)), settings=test_settings, net=_FakeNet(ramp))
    assert matrix.method == "mono"
    assert matrix.coverage == 1.0
    flat = [value for row in matrix.values for value in row]
    assert all(0.0 <= value <= 1.0 for value in flat)
    assert min(flat) == 0.0
    assert matrix.values[0][0] < matrix.values[-1][-1]


def test_estimate_depth_keeps_aspect_with_letterbox(test_settings: Settings) -> None:
    """A wide image keeps its aspect after the square model pass."""
    plain = np.full((1, 8, 8), 5.0, dtype=np.float32)
    matrix = estimate_depth(Image.new("RGB", (16, 8)), settings=test_settings, net=_FakeNet(plain))
    assert (matrix.width, matrix.height) == (16, 8)
    assert all(value == 0.0 for row in matrix.values for value in row)


def test_estimate_depth_wraps_broken_net(test_settings: Settings) -> None:
    """A failing network is a plain-language error."""

    class _Broken:
        """Network stub failing on input."""

        def setInput(self, blob: object) -> None:
            """Fail as a broken backend would."""
            raise RuntimeError("boom")

        def forward(self) -> np.ndarray:
            """Never reached."""
            raise AssertionError("unreachable")

    with pytest.raises(MonoDepthError, match="impossible"):
        estimate_depth(Image.new("RGB", (8, 8)), settings=test_settings, net=_Broken())


def test_model_path_download_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An offline download is a plain-language error."""
    settings = Settings(baseten_api_key="key", data_dir=tmp_path)

    def _offline(url: str, target: str) -> None:
        """Fail as an offline network would."""
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlretrieve", _offline)
    with pytest.raises(MonoDepthError, match="téléchargement impossible"):
        model_path(settings)


def test_model_path_rejects_bad_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupted download is deleted with a plain-language error."""
    settings = Settings(baseten_api_key="key", data_dir=tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlretrieve", lambda url, target: Path(target).write_bytes(b"junk")
    )
    with pytest.raises(MonoDepthError, match="corrompu"):
        model_path(settings)
    assert not (tmp_path / settings.monodepth_model_file).exists()


def test_mono_depth_command_estimates_photo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    photo_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """mono-depth prints the AI method and saves the GIF (offline)."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    canned = DepthMatrix(
        width=8,
        height=8,
        values=tuple(tuple(0.5 for _ in range(8)) for _ in range(8)),
        coverage=1.0,
        left_index=-1,
        right_index=-1,
        method="mono",
    )
    monkeypatch.setattr("temoin.__main__.estimate_depth", lambda *args, **kwargs: canned)
    wiggle = tmp_path / "mono.gif"
    assert main(["mono-depth", str(photo_path), "--wiggle", str(wiggle)]) == 0
    out = capsys.readouterr().out
    assert "IA monoculaire" in out
    assert wiggle.is_file()


def test_mono_depth_command_rejects_missing_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing mono video keeps the video error contract."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    assert main(["mono-depth", str(tmp_path / "nope.mp4")]) == 1
    assert "introuvable" in capsys.readouterr().err
