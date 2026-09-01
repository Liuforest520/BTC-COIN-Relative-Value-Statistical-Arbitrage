from pathlib import Path


def test_config_file_exists():
    assert Path("config/config.yaml").exists()
