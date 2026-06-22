import importlib.util
from pathlib import Path

import numpy as np


def load_animation_module():
  repo_root = Path(__file__).resolve().parents[1]
  scripts_dir = repo_root / "scripts"
  module_path = scripts_dir / "animate_pose_graph_iterations.py"
  spec = importlib.util.spec_from_file_location(
      "animate_pose_graph_iterations", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  import sys
  sys.path.insert(0, str(scripts_dir))
  sys.modules[spec.name] = module
  try:
    spec.loader.exec_module(module)
  finally:
    sys.path.remove(str(scripts_dir))
    sys.modules.pop(spec.name, None)
  return module


def write_pose_matrix(path: Path, x: float):
  dense = np.array([[1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
  np.savetxt(path, dense)


def test_load_stage_frames_ignores_unlisted_stale_files_when_metadata_exists(tmp_path):
  module = load_animation_module()
  write_pose_matrix(tmp_path / "init_0000.txt", 0.0)
  write_pose_matrix(tmp_path / "init_0001.txt", 1.0)
  write_pose_matrix(tmp_path / "init_0002.txt", 2.0)
  (tmp_path / "init_metadata.csv").write_text(
      "frame,stage,pcg_iter,residual_norm\n"
      "0,D-CCI-PCG translation solve,0,10\n"
      "1,D-CCI-PCG translation solve,5,1\n",
      encoding="utf-8")

  frames = module.load_stage_frames(
      tmp_path, "DRAN", None, "D-CCI-PCG initialization", "init", False)

  assert [frame.step for frame in frames] == [0, 5]
  assert [frame.stage for frame in frames] == [
      "D-CCI-PCG translation solve",
      "D-CCI-PCG translation solve",
  ]
  assert module.format_stage_metadata(frames[0].metadata) == (
      "D-CCI-PCG iter: 0\nD-CCI residual: 1.000e+01")
  assert module.format_stage_metadata(frames[1].metadata) == (
      "D-CCI-PCG iter: 5\nD-CCI residual: 1.000e+00")
