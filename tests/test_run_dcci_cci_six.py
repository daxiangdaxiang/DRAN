import importlib.util
from pathlib import Path


def load_runner_module():
  repo_root = Path(__file__).resolve().parents[1]
  module_path = repo_root / "scripts" / "run_dcci_cci_six.py"
  spec = importlib.util.spec_from_file_location("run_dcci_cci_six", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_default_dcci_cci_datasets_include_ais2klinik():
  module = load_runner_module()

  datasets = dict(module.DEFAULT_DATASETS)

  assert datasets["ais2klinik"] == "data/ais2klinik.g2o"


def test_runner_accepts_initialization_mode_override():
  module = load_runner_module()

  default_args = module.parse_args([])
  auto_args = module.parse_args(["--initialization-mode", "ted_cci_sr_auto"])

  assert default_args.initialization_mode == "dpcg_cci"
  assert auto_args.initialization_mode == "ted_cci_sr_auto"
