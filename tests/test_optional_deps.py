"""Optional heavy dependencies degrade gracefully instead of crashing imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from temoin import forensics, meta_api, monodepth, relief, stereo, video
from temoin.config import Settings
from temoin.images import ImageError
from temoin.meta_api import MetaAPIError
from temoin.monodepth import MonoDepthError
from temoin.relief import ReliefError
from temoin.stereo import DepthMatrix, StereoError
from temoin.video import VideoError


def test_video_needs_opencv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """extract_frames explains the missing OpenCV instead of crashing."""
    monkeypatch.setattr(video, "cv2", None)
    with pytest.raises(VideoError, match="OpenCV"):
        video.extract_frames(
            tmp_path / "tour.mp4", max_frames=2, max_pixels=64, max_megabytes=200
        )


def test_stereo_compute_needs_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stereo measurement explains missing OpenCV/NumPy instead of crashing."""
    monkeypatch.setattr(stereo, "cv2", None)
    image = Image.new("RGB", (8, 8))
    with pytest.raises(StereoError, match="OpenCV"):
        stereo.compute_depth(image, image)


def test_stereo_cloud_needs_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 3D cloud explains missing Plotly instead of crashing."""
    monkeypatch.setattr(stereo, "go", None)
    matrix = DepthMatrix(
        width=2,
        height=2,
        values=((0.1, 0.2), (0.3, 0.4)),
        coverage=1.0,
        left_index=0,
        right_index=1,
    )
    with pytest.raises(StereoError, match="Plotly"):
        stereo.relief_cloud(Image.new("RGB", (2, 2)), matrix)


def test_monodepth_needs_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    """AI relief explains missing OpenCV instead of crashing."""
    monkeypatch.setattr(monodepth, "cv2", None)
    with pytest.raises(MonoDepthError, match="OpenCV"):
        monodepth.estimate_depth(Image.new("RGB", (8, 8)))


def test_forensics_needs_vision(monkeypatch: pytest.MonkeyPatch, photo_path: Path) -> None:
    """Photo checks explain missing OpenCV instead of crashing."""
    monkeypatch.setattr(forensics, "cv2", None)
    with pytest.raises(ImageError, match="OpenCV"):
        forensics.analyze_photo(photo_path)


def test_segmentation_needs_parser(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    """SAM segmentation explains the missing parser instead of crashing."""
    monkeypatch.setattr(meta_api, "parse_responses_stream", None)
    with pytest.raises(MetaAPIError, match="meta-sam-parser"):
        meta_api.segment_image(
            "data:image/jpeg;base64,xx", "headphones", settings=test_settings
        )


def test_relief_needs_numpy(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    """Raking relief explains missing NumPy instead of crashing."""
    monkeypatch.setattr(relief, "np", None)
    with pytest.raises(ReliefError, match="NumPy"):
        relief.run_relief([], settings=test_settings)


def test_app_boots_without_heavy_modules() -> None:
    """The Streamlit app boots with cv2/plotly/sam-parser/openai all missing."""
    repo = Path(__file__).resolve().parent.parent
    script = (
        "import sys\n"
        "for name in (\n"
        "    'cv2', 'plotly', 'plotly.io', 'plotly.graph_objects',\n"
        "    'meta_sam_parser', 'openai',\n"
        "):\n"
        "    sys.modules[name] = None\n"
        "from streamlit.testing.v1 import AppTest\n"
        "app = AppTest.from_file('app.py')\n"
        "app.run()\n"
        "assert not app.exception, app.exception\n"
        "assert len(app.tabs) == 6\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
