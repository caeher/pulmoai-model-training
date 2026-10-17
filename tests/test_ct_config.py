import subprocess
import sys
from pathlib import Path

import pytest

from pulmoai.config import ConfigError, config_from_dict, load_config


def test_ct_config_roundtrip(config_dir):
    config = load_config(config_dir)
    assert config_from_dict(config.to_dict()) == config
    assert config_from_dict(config.to_dict()).fingerprint() == config.fingerprint()
    assert config.data.dataset == "luna16"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("data", "spacing_xyz", [1, 0, 1]),
        ("data", "patch_size", 7),
        ("data", "patch_size", 16.0),
        ("data", "hu_window", [400, -1000]),
        ("train", "lr", float("nan")),
        ("train", "epochs", True),
        ("train", "amp", "false"),
        ("train", "device", "tpu"),
        ("model", "base_channels", 3),
        ("evaluation", "overlap", 1),
        ("tracking", "output_dir", "../private"),
        ("data", "unknown", 1),
        ("seeds", "global_seed", -1),
        ("data", "split_fractions", [0.5, 0.5]),
    ],
)
def test_invalid_ct_config(section, key, value):
    with pytest.raises(ConfigError):
        config_from_dict({section: {key: value}})


def test_lightweight_import_and_help():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pulmoai, pulmoai.config, sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'SimpleITK' not in sys.modules",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
