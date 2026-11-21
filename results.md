```
== Placement: three devices, one memory limit ==
  assignment: {'fa': 0, 'fb': 1, 'fc': 2, 'ra': 0, 'rb': 1, 'rc': 2, 'tot': 0}
  copies inserted: [('copy', 'rb', 1, 0), ('copy', 'rc', 2, 0)]
  planned bytes moved: 160 | executed bytes moved: 160 | memory per device: [3520, 3280, 3280] | executed: [3520, 3280, 3280]
  result matches numpy: True
== A sharded training step: linear model, 48 examples, 6 features, 2 outputs ==
  loss on 4 devices: 4.668106 -> 0.111985 after 20 steps
  1 device(s): setup moved 0 bytes | per step moved 0, computed on 12216 (0.0000 moved per byte computed) | replicas identical: True | equals the 1-device weights within 1e-9: True
  2 device(s): setup moved 1632 bytes | per step moved 200, computed on 12808 (0.0156 moved per byte computed) | replicas identical: True | equals the 1-device weights within 1e-9: True
  4 device(s): setup moved 2592 bytes | per step moved 600, computed on 13992 (0.0429 moved per byte computed) | replicas identical: True | equals the 1-device weights within 1e-9: True
  sharded runs equal the single-device run within 1e-9: True
  4 devices, whole run: bytes moved 14592, bytes computed on 279840, copies 189, kernels 600, peak memory per device [3168, 1640, 1640, 1640]
```
