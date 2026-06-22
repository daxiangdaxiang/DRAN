import importlib.util
from pathlib import Path

import pytest


def load_renderer_module():
  repo_root = Path(__file__).resolve().parents[1]
  module_path = repo_root / "scripts" / "render_dcci_pcg_process_videos.py"
  spec = importlib.util.spec_from_file_location(
      "render_dcci_pcg_process_videos", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def write_pose(path: Path):
  path.write_text(
      "1 0 0 0\n"
      "0 1 0 0\n"
      "0 0 1 0\n",
      encoding="utf-8")


def test_dcci_process_trace_requires_metadata(tmp_path):
  module = load_renderer_module()
  init_dir = tmp_path / "initialization_estimates"
  init_dir.mkdir()
  write_pose(init_dir / "init_0000.txt")

  with pytest.raises(FileNotFoundError, match="D-CCI-PCG process metadata"):
    module.validate_dcci_process_trace(init_dir)


def test_dcci_process_trace_rejects_endpoint_only_trace(tmp_path):
  module = load_renderer_module()
  init_dir = tmp_path / "initialization_estimates"
  init_dir.mkdir()
  write_pose(init_dir / "init_0000.txt")
  (init_dir / "init_metadata.csv").write_text(
      "frame,stage,pcg_iter,residual_norm\n"
      "0,D-CCI-PCG translation solve,0,10\n",
      encoding="utf-8")

  with pytest.raises(ValueError, match="at least two"):
    module.validate_dcci_process_trace(init_dir)


def test_dcci_process_trace_reports_valid_multiframe_trace(tmp_path):
  module = load_renderer_module()
  init_dir = tmp_path / "initialization_estimates"
  init_dir.mkdir()
  write_pose(init_dir / "init_0000.txt")
  write_pose(init_dir / "init_0001.txt")
  (init_dir / "init_metadata.csv").write_text(
      "frame,stage,pcg_iter,residual_norm\n"
      "0,D-CCI-PCG translation solve,0,10\n"
      "1,D-CCI-PCG translation solve,8,1\n",
      encoding="utf-8")

  summary = module.validate_dcci_process_trace(init_dir)

  assert summary["metadata_rows"] == 2
  assert summary["pose_frames"] == 2
  assert summary["first_pcg_iter"] == "0"
  assert summary["last_pcg_iter"] == "8"
