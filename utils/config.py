from pathlib import Path
from konfik import Konfik

BASE_DIR = Path(__file__).parents[1]
CONFIG_PATH_TOML = BASE_DIR / "config.toml"
konfik = Konfik(config_path=CONFIG_PATH_TOML)
config = konfik.config