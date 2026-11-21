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

# Step 2 - Cluster
class Cluster:
    """n simulated devices and the counted copies between them."""

    def __init__(self, n, mem_limit):
        # Create n devices, each with the same memory limit.
        self.devices = [
            Device(f"gpu{i}", mem_limit)
            for i in range(n)
        ]

        # Initialize cross-device communication counters.
        self.bytes_moved = 0
        self.copies = 0

        # Store transfer records as (src, dst, name, nbytes).
        self.log = []

    def copy(self, src, dst, name, dst_name=None, region=None):
        """Copy a buffer (or a region of it) from device src to device dst."""

        # Get the source and destination devices.
        src_device = self.devices[src]
        dst_device = self.devices[dst]

        # Retrieve the source buffer. A missing name raises KeyError.
        source_array = src_device.buffers[name]

        # Extract only the requested region when specified.
        if region is not None:
            source_array = source_array[region]

        # Use the original buffer name unless another destination
        # buffer name has been explicitly provided.
        if dst_name is None:
            dst_name = name

        # Allocate the copied data on the destination device.
        # Device.alloc handles float64 copying and memory-limit checks.
        dst_device.alloc(dst_name, source_array)

        # Determine the actual number of bytes stored on the destination.
        nbytes = dst_device.buffers[dst_name].nbytes

        # Only cross-device copies contribute to communication statistics.
        if src != dst:
            self.bytes_moved += nbytes
            self.copies += 1

            # Record the transfer using the original source buffer name.
            self.log.append((src, dst, name, nbytes))

        # Return the number of bytes stored on the destination device.
        return nbytes

    def stats(self):
        """Return bytes_moved, copies, bytes_computed, kernels and peak."""

        # Aggregate computation statistics across all devices.
        total_bytes_computed = sum(
            device.bytes_computed
            for device in self.devices
        )

        total_kernels = sum(
            device.kernels
            for device in self.devices
        )

        # Return communication and computation statistics.
        return {
            "bytes_moved": self.bytes_moved,
            "copies": self.copies,
            "bytes_computed": total_bytes_computed,
            "kernels": total_kernels,
            "peak": [
                device.peak
                for device in self.devices
            ]
        }

# Step 3 - shard_bounds
def shard_bounds(n, k):
    """Split 0..n-1 into k contiguous shards whose sizes differ by at most one."""

    # Calculate the base shard size and the number of extra elements.
    q, r = divmod(n, k)

    bounds = []
    start = 0

    # Assign one shard to each device.
    for i in range(k):

        # The first r shards receive one additional element.
        shard_size = q + (1 if i < r else 0)

        # Calculate the exclusive ending index of the shard.
        stop = start + shard_size

        # Store the shard boundaries as (start, stop).
        bounds.append((start, stop))

        # The next shard begins where the current shard ends.
        start = stop

    return bounds

# Step 4 - scatter
def handle(name, shape, layout, axis=None, bounds=None, device=None):
    return {
        "name": name,
        "shape": tuple(shape),
        "layout": layout,
        "axis": axis,
        "bounds": bounds,
        "device": device
    }


def axis_slice(ndim, axis, start, stop):
    return tuple(
        slice(start, stop) if d == axis else slice(None)
        for d in range(ndim)
    )


def scatter(cluster, name, axis=0, home=0):
    """Cut the buffer on device home into one shard per device and return a sharded handle."""

    # Get the number of devices and normalize the home device index.
    num_devices = len(cluster.devices)

    home_index = home
    if home_index < 0:
        home_index += num_devices

    if not 0 <= home_index < num_devices:
        raise IndexError("Home device index out of range")

    # Retrieve the complete tensor from the home device.
    source = cluster.devices[home_index].buffers[name]

    # Preserve the original tensor shape.
    shape = tuple(source.shape)
    ndim = source.ndim

    # Normalize the axis to support negative axis indices.
    axis_index = axis + ndim if axis < 0 else axis

    if not 0 <= axis_index < ndim:
        raise ValueError("Axis out of range")

    # Calculate the boundaries for all device shards.
    bounds = shard_bounds(shape[axis_index], num_devices)

    # Send each non-home shard to its corresponding device.
    # The source must remain whole until all remote copies finish.
    for device_index, (start, stop) in enumerate(bounds):

        if device_index == home_index:
            continue

        # Construct a slice selecting only this device's shard.
        region = axis_slice(ndim, axis_index, start, stop)

        # Copy the selected region to the destination device.
        # Empty shards are copied as well.
        cluster.copy(
            home_index,
            device_index,
            name,
            region=region
        )

    # Reduce the home device's buffer to its own shard.
    # This is a same-device copy and does not count as communication.
    start, stop = bounds[home_index]

    home_region = axis_slice(ndim, axis_index, start, stop)

    cluster.copy(
        home_index,
        home_index,
        name,
        region=home_region
    )

    # Return the metadata handle for the sharded tensor.
    return handle(
        name,
        shape,
        "sharded",
        axis=axis,
        bounds=bounds
    )


def replicate(cluster, name, home=0):
    """Copy the buffer on device home to every other device and return a replicated handle."""

    # Get the number of devices and normalize the home index.
    num_devices = len(cluster.devices)

    home_index = home
    if home_index < 0:
        home_index += num_devices

    if not 0 <= home_index < num_devices:
        raise IndexError("Home device index out of range")

    # Retrieve the complete tensor from the home device.
    source = cluster.devices[home_index].buffers[name]

    # Preserve the original tensor shape.
    shape = tuple(source.shape)

    # Copy the complete tensor to every other device in device order.
    for device_index in range(num_devices):

        if device_index == home_index:
            continue

        cluster.copy(
            home_index,
            device_index,
            name
        )

    # Return the metadata handle for the replicated tensor.
    return handle(
        name,
        shape,
        "replicated"
    )

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

