"""
Many Devices: Sharding and Placement for a Tensor Compiler scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

from solution import *  # noqa: F401,F403

import numpy as np

print("== Placement: three devices, one memory limit ==")
inputs = {"a": {"device": 0, "bytes": 1600}, "b": {"device": 1, "bytes": 1600}, "c": {"device": 2, "bytes": 1600}}
kernels = [
    {"name": "fa", "inputs": ["a"], "bytes": 1600, "fn": lambda a: a * 2.0},
    {"name": "fb", "inputs": ["b"], "bytes": 1600, "fn": lambda b: b + 1.0},
    {"name": "fc", "inputs": ["c"], "bytes": 1600, "fn": lambda c: c * c},
    {"name": "ra", "inputs": ["fa"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "rb", "inputs": ["fb"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "rc", "inputs": ["fc"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "tot", "inputs": ["ra", "rb", "rc"], "bytes": 80, "fn": lambda p, q, r: p + q + r},
]
plan = place_graph(kernels, inputs, 3, 4000)
rng = np.random.default_rng(1)
data = {name: rng.standard_normal((20, 10)) for name in inputs}
cluster = Cluster(3, 4000)
for name, info in inputs.items():
    cluster.devices[info["device"]].alloc(name, data[name])
fns = {k["name"]: k for k in kernels}
for item in plan["schedule"]:
    if item[0] == "copy":
        cluster.copy(item[2], item[3], item[1])
    else:
        cluster.devices[item[2]].run(item[1], fns[item[1]]["fn"], *fns[item[1]]["inputs"])
ref = (data["a"] * 2.0).sum(axis=0) + (data["b"] + 1.0).sum(axis=0) + (data["c"] * data["c"]).sum(axis=0)
got = cluster.devices[plan["assignment"]["tot"]].buffers["tot"]
print("  assignment:", plan["assignment"])
print("  copies inserted:", [s for s in plan["schedule"] if s[0] == "copy"])
print("  planned bytes moved:", plan["bytes_moved"], "| executed bytes moved:", cluster.bytes_moved,
      "| memory per device:", plan["memory"], "| executed:", [d.used for d in cluster.devices])
print("  result matches numpy:", bool(np.allclose(got, ref, rtol=0, atol=1e-9)))

print("== A sharded training step: linear model, 48 examples, 6 features, 2 outputs ==")
N, D, M, STEPS, LR = 48, 6, 2, 20, 0.1
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D))
Y = X @ rng.standard_normal((D, M)) + 0.05 * rng.standard_normal((N, M))


def fit(k):
    c = Cluster(k, 16384)
    c.devices[0].alloc("x", X)
    c.devices[0].alloc("y", Y)
    c.devices[0].alloc("w", np.zeros((D, M)))
    x, y, w = scatter(c, "x"), scatter(c, "y"), replicate(c, "w")
    setup = c.stats()
    losses = [train_step(c, x, y, w, LR) for _ in range(STEPS)]
    return c, setup, losses


runs = {k: fit(k) for k in (1, 2, 4)}
w1 = runs[1][0].devices[0].buffers["w"]
print(f"  loss on 4 devices: {runs[4][2][0]:.6f} -> {runs[4][2][-1]:.6f} after {STEPS} steps")
for k in (1, 2, 4):
    c, setup, losses = runs[k]
    s = c.stats()
    wk = c.devices[0].buffers["w"]
    same = len({d.buffers["w"].tobytes() for d in c.devices}) == 1
    step_moved = (s["bytes_moved"] - setup["bytes_moved"]) // STEPS
    step_computed = (s["bytes_computed"] - setup["bytes_computed"]) // STEPS
    print(f"  {k} device(s): setup moved {setup['bytes_moved']} bytes | per step moved {step_moved}, computed on {step_computed}"
          f" ({step_moved / step_computed:.4f} moved per byte computed) | replicas identical: {same}"
          f" | equals the 1-device weights within 1e-9: {float(np.abs(wk - w1).max()) <= 1e-9}")
ok = all(float(np.abs(runs[k][0].devices[0].buffers["w"] - w1).max()) <= 1e-9 for k in (2, 4))
print("  sharded runs equal the single-device run within 1e-9:", ok)
t = runs[4][0].stats()
print(f"  4 devices, whole run: bytes moved {t['bytes_moved']}, bytes computed on {t['bytes_computed']},"
      f" copies {t['copies']}, kernels {t['kernels']}, peak memory per device {t['peak']}")

