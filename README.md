# Many Devices: Sharding and Placement for a Tensor Compiler

Week 10 of the Compilers for Machine Learning course is one line: multiple GPUs and sharding, scale execution across multiple GPUs. This project builds the back end that line asks for. A tensor is cut into shards, one per device; each device runs the same kernel on its own shard; and every byte that has to cross from one device to another is a copy the compiler inserts and you can count. You build a simulated device with a memory limit, a cluster whose copies are logged, the split rule and its inverse, re-sharding when two operands disagree, sharded elementwise kernels, reductions and matmul, a placer that puts a kernel graph on devices with the fewest transferred bytes under a memory limit, and an all-reduce written out as copies. The last step is a full data-parallel training step for a linear model, checked against the same step on one device, with a report of bytes moved against bytes computed on. The devices are simulated in plain Python and numpy, so every count is exact and every step is testable. The course publishes no rules for this week, so the rules here (the split rule, what a copy costs, where a reduction is combined) are one reasonable design, not the course's specification.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** Device
- [x] **2.** Cluster
- [x] **3.** shard_bounds
- [x] **4.** scatter
- [x] **5.** gather
- [x] **6.** reshard
- [x] **7.** map_shards
- [x] **8.** reduce_shards
- [x] **9.** matmul_shards
- [x] **10.** place_graph
- [x] **11.** allreduce_sum
- [x] **12.** train_step

## Results

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
``
