"""
Many Devices: Sharding and Placement for a Tensor Compiler

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - Device
import numpy as np


class Device:
    """One simulated device: named float64 buffers, a memory limit and counters."""

    def __init__(self, name, mem_limit):
        # Device identification and memory capacity.
        self.name = name
        self.mem_limit = mem_limit

        # Store allocated buffers by their names.
        self.buffers = {}

        # Initialize memory usage and performance counters.
        self.used = 0
        self.peak = 0
        self.kernels = 0
        self.bytes_computed = 0

    def alloc(self, name, array):
        """Store a float64 copy of array under name, or raise MemoryError."""

        # Create an independent float64 copy of the input array.
        arr = np.array(array, dtype=np.float64, copy=True)

        # Account for the existing buffer if the name is already allocated.
        old_size = self.buffers[name].nbytes if name in self.buffers else 0

        # Calculate memory usage after replacing the existing buffer.
        new_used = self.used - old_size + arr.nbytes

        # Reject the allocation without modifying the existing state.
        if new_used > self.mem_limit:
            raise MemoryError(
                f"Device '{self.name}' memory limit exceeded: "
                f"{new_used} > {self.mem_limit} bytes"
            )

        # Store the new buffer and update memory counters.
        self.buffers[name] = arr
        self.used = new_used
        self.peak = max(self.peak, self.used)

    def free(self, name):
        """Remove a buffer and give its bytes back."""

        # Remove the buffer. pop() raises KeyError if the name is missing.
        arr = self.buffers.pop(name)

        # Release the memory occupied by the removed buffer.
        self.used -= arr.nbytes

    def run(self, out, fn, *inputs):
        """Run one kernel on this device's buffers and store its result under out."""

        # Retrieve input buffers. Missing names automatically raise KeyError.
        input_arrays = [self.buffers[name] for name in inputs]

        # Calculate the total number of bytes read by this kernel.
        # Count repeated inputs separately because each is passed to the kernel.
        bytes_read = sum(arr.nbytes for arr in input_arrays)

        # Execute the kernel function on the input arrays.
        result = fn(*input_arrays)

        # Allocate the output buffer. If allocation fails, counters remain unchanged.
        self.alloc(out, result)

        # Calculate the bytes written by the output buffer.
        bytes_written = self.buffers[out].nbytes

        # Update execution and computation counters after successful execution.
        self.kernels += 1
        self.bytes_computed += bytes_read + bytes_written

# Step 2 - Cluster (not yet solved)
# TODO: implement

# Step 3 - shard_bounds (not yet solved)
# TODO: implement

# Step 4 - scatter (not yet solved)
# TODO: implement

# Step 5 - gather (not yet solved)
# TODO: implement

# Step 6 - reshard (not yet solved)
# TODO: implement

# Step 7 - map_shards (not yet solved)
# TODO: implement

# Step 8 - reduce_shards (not yet solved)
# TODO: implement

# Step 9 - matmul_shards (not yet solved)
# TODO: implement

# Step 10 - place_graph (not yet solved)
# TODO: implement

# Step 11 - allreduce_sum (not yet solved)
# TODO: implement

# Step 12 - train_step (not yet solved)
# TODO: implement

