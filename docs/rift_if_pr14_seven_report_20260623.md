# RIFT-IF PR14 Seven-Dataset Report

Command:

```sh
timeout 3600 python scripts/run_rift_if_seven.py \
  --bench-bin build/bin/bench-dcci \
  --output-dir results/rift_if_pr14_seven_20260623 \
  --num-robots 5 \
  --max-iters 20000 \
  --rel-tol 1e-10 \
  --abs-tol 1e-10
```

Artifacts:

- `results/rift_if_pr14_seven_20260623/summary.csv`
- `results/rift_if_pr14_seven_20260623/summary.md`
- `results/rift_if_pr14_seven_20260623/report.json`
- `results/rift_if_pr14_seven_20260623/commands.txt`

All seven RIFT deployment runs finished with
`used_global_matrix=0`, `used_direct_solver=0`, and `used_collective=0`.
`rift_auto` selected `rift_cak` on every real dataset. This means the current
exact symbolic path is still too restrictive for these full graphs: either the
exact message gate rejects the tree or the symbolic tree does not satisfy the
running-intersection check, so the stable-network fallback is used.

| Dataset | RIFT backend | Pose diff vs CCI | Cost gap | RIFT comm MB | Direct TED comm MB | DPCG comm MB | RIFT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parking-garage | `rift_cak` | `2.038519898203906e-08` | `1.462615784042498e-08` | `0.523681640625` | `1.925777435302734` | `0.2054595947265625` | `82881.617063` |
| sphere | `rift_cak` | `8.900815740589735e-13` | `7.190465112216771e-09` | `0.0751953125` | `15.29473495483398` | `0.08152008056640625` | `8604.695844` |
| torus | `rift_cak` | `7.914702130656033e-12` | `1.515945768915117e-08` | `0.08551025390625` | `34.26914215087891` | `0.08563995361328125` | `73987.156353` |
| CSAIL | `rift_cak` | `1.403155918732662e-14` | `4.689582056016661e-13` | `0.05889892578125` | `0.06382369995117188` | `0.1897811889648438` | `90.26889300000001` |
| inter | `rift_cak` | `1.157534968110387e-13` | `1.13686837721616e-13` | `0.0504150390625` | `0.06888198852539062` | `0.08060455322265625` | `131.992914` |
| manhattan | `rift_cak` | `9.497554877566221e-13` | `1.164551122201374e-09` | `0.09710693359375` | `2.538082122802734` | `0.145263671875` | `6414.505825` |
| ais2klinik | `rift_cak` | `6.802305405357172e-13` | `4.2842884795391e-10` | `0.49102783203125` | `0.9659080505371094` | `1.564216613769531` | `304999.053011` |

Interpretation:

- Accuracy: RIFT-IF-CAK matches centralized CCI closely on all seven datasets.
  The largest observed pose-matrix difference is about `2.04e-08` on
  `parking-garage`; the other six are around `1e-12` to `1e-14`.
- Communication: RIFT-IF reports lower MB than direct TED on all seven
  datasets, and is close to or below DPCG on six of seven. `parking-garage`
  remains higher than DPCG in MB.
- Runtime: current CAK fallback is not yet compute-efficient on large graphs.
  The most severe case is `ais2klinik`, where RIFT takes about `305 s` versus
  about `5.1 s` for direct TED and `16.3 s` for DPCG. This is the main
  remaining engineering/research bottleneck after PR14.

Next technical implication:

The PR14 benchmark support is functional and guarded, but the exact symbolic
backend is not yet the dominant full-dataset path. The next RIFT-IF work should
focus on improving the symbolic clique-tree construction/running-intersection
robustness and on preconditioned/pipelined CAK, because the fallback now gives
the desired numerical equivalence and communication accounting but is too slow
for the largest graphs.
