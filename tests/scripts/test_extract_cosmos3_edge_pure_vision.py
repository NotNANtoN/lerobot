from pathlib import Path

import pytest

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from scripts.video_vam import extract_cosmos3_edge_pure_vision as builder


def test_v1_revision_keeps_published_default(tmp_path: Path):
    args = builder.parse_args(["--output-dir", str(tmp_path / "cache")])
    assert args.dataset_revision == CUBE_OUT_OF_BOX_CONTRACT.revision


def test_non_v1_requires_explicit_revision(tmp_path: Path):
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                "--output-dir",
                str(tmp_path / "cache"),
                "--dataset-repo-id",
                "Orellius/cube_out_of_box_v2",
            ]
        )


def test_non_v1_retains_its_own_revision(tmp_path: Path):
    args = builder.parse_args(
        [
            "--output-dir",
            str(tmp_path / "cache"),
            "--dataset-repo-id",
            "Orellius/cube_out_of_box_v2",
            "--dataset-root",
            str(tmp_path / "v2"),
            "--dataset-revision",
            "5d0325cc1412f4774223a0beb528958108814962",
        ]
    )
    assert args.dataset_revision == "5d0325cc1412f4774223a0beb528958108814962"
    assert args.dataset_revision != CUBE_OUT_OF_BOX_CONTRACT.revision
