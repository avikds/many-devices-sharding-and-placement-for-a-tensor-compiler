# Many Devices: Sharding and Placement for a Tensor Compiler

Week 10 of the Compilers for Machine Learning course is one line: multiple GPUs and sharding, scale execution across multiple GPUs. This project builds the back end that line asks for. A tensor is cut into shards, one per device; each device runs the same kernel on its own shard; and every byte that has to cross from one device to another is a copy the compiler inserts and you can count. You build a simulated device with a memory limit, a cluster whose copies are logged, the split rule and its inverse, re-sharding when two operands disagree, sharded elementwise kernels, reductions and matmul, a placer that puts a kernel graph on devices with the fewest transferred bytes under a memory limit, and an all-reduce written out as copies. The last step is a full data-parallel training step for a linear model, checked against the same step on one device, with a report of bytes moved against bytes computed on. The devices are simulated in plain Python and numpy, so every count is exact and every step is testable. The course publishes no rules for this week, so the rules here (the split rule, what a copy costs, where a reduction is combined) are one reasonable design, not the course's specification.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** Device
- [x] **2.** Cluster
- [ ] **3.** shard_bounds
- [ ] **4.** scatter
- [ ] **5.** gather
- [ ] **6.** reshard
- [ ] **7.** map_shards
- [ ] **8.** reduce_shards
- [ ] **9.** matmul_shards
- [ ] **10.** place_graph
- [ ] **11.** allreduce_sum
- [ ] **12.** train_step

---

Built on Deep-ML.
