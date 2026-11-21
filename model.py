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

# Step 5 - gather
def gather(cluster, t, home=0):
    """Bring the tensor behind handle t back to device home as one buffer; update and return t."""

    # Normalize and validate the home device index.
    num_devices = len(cluster.devices)

    if home < 0:
        home += num_devices

    if not 0 <= home < num_devices:
        raise IndexError("Home device index out of range")

    name = t["name"]
    layout = t["layout"]

    # Get the destination device.
    home_device = cluster.devices[home]

    if layout == "sharded":

        axis = t["axis"]

        # Transfer every remote shard to the home device.
        # Each remote shard is stored under a temporary name.
        # Transfers are performed in device order.
        for i in range(num_devices):

            if i == home:
                continue

            temp_name = f"{name}@{i}"

            # Copy this device's shard to the home device.
            cluster.copy(
                i,
                home,
                name,
                dst_name=temp_name
            )

            # Free the original shard after a successful transfer.
            cluster.devices[i].free(name)

        # Construct input names in device order.
        # The home device's own shard remains under the original name.
        input_names = [
            name if i == home else f"{name}@{i}"
            for i in range(num_devices)
        ]

        # Concatenate all shards along the original sharding axis.
        # The result replaces the home device's original shard.
        home_device.run(
            name,
            lambda *arrays: np.concatenate(arrays, axis=axis),
            *input_names
        )

        # Free the temporary shard buffers after reconstruction.
        for i in range(num_devices):
            if i != home:
                home_device.free(f"{name}@{i}")

    elif layout == "replicated":

        # Keep the complete tensor on the home device.
        # Free every other device's replica.
        for i in range(num_devices):

            if i == home:
                continue

            cluster.devices[i].free(name)

    elif layout == "single":

        # Find the device currently holding the complete tensor.
        current_device = t["device"]

        # Move the tensor only when it is not already on the home device.
        if current_device != home:

            # Copy the complete tensor to the home device.
            cluster.copy(
                current_device,
                home,
                name
            )

            # Release the buffer on its previous device.
            cluster.devices[current_device].free(name)

    else:
        raise ValueError(f"Unsupported tensor layout: {layout}")

    # Update the existing handle to represent a single-device tensor.
    t.update({
        "layout": "single",
        "axis": None,
        "bounds": None,
        "device": home
    })

    # Return the updated handle.
    return t

# Step 6 - reshard
def reshard(cluster, t, axis, bounds=None):
    """Recut the sharded tensor behind t along axis with the given bounds; update and return t."""

    # Re-sharding is only supported for sharded tensors.
    if t["layout"] != "sharded":
        raise ValueError("Only sharded tensors can be re-sharded")

    num_devices = len(cluster.devices)
    name = t["name"]
    shape = tuple(t["shape"])
    ndim = len(shape)

    # Normalize both axes so negative axis indices work correctly.
    old_axis = t["axis"] % ndim
    new_axis = axis % ndim

    if not 0 <= new_axis < ndim:
        raise ValueError("Axis out of range")

    # Calculate the target shard boundaries if none were supplied.
    if bounds is None:
        target_bounds = shard_bounds(shape[new_axis], num_devices)
    else:
        target_bounds = [tuple(b) for b in bounds]

    if len(target_bounds) != num_devices:
        raise ValueError("Bounds must contain one entry per device")

    # Normalize the existing boundaries for comparison.
    old_bounds = [tuple(b) for b in t["bounds"]]

    # Avoid unnecessary copies and kernel executions when the layout
    # already matches the requested layout.
    if old_axis == new_axis and old_bounds == target_bounds:
        return t

    # Each destination stores the names of the blocks it receives.
    received_blocks = [[] for _ in range(num_devices)]

    # Process every target device, then every source device, in order.
    for j in range(num_devices):

        target_start, target_stop = target_bounds[j]

        for i in range(num_devices):

            source_start, source_stop = old_bounds[i]

            # Determine the region to extract from the source shard.
            if old_axis == new_axis:

                # For the same axis, extract only the overlap between
                # the source shard and the target shard.
                overlap_start = max(source_start, target_start)
                overlap_stop = min(source_stop, target_stop)

                # No overlap means there is nothing to transfer.
                if overlap_start >= overlap_stop:
                    continue

                # Convert global indices into source-shard-local indices.
                local_start = overlap_start - source_start
                local_stop = overlap_stop - source_start

                region = axis_slice(
                    ndim,
                    old_axis,
                    local_start,
                    local_stop
                )

            else:

                # For different axes, restrict the source shard to
                # the target interval along the new axis.
                region = axis_slice(
                    ndim,
                    new_axis,
                    target_start,
                    target_stop
                )

            # Check whether the selected block contains any elements.
            source_array = cluster.devices[i].buffers[name]
            block = source_array[region]

            # Empty blocks are skipped.
            if block.size == 0:
                continue

            # Use a temporary name identifying the source device.
            temp_name = f"{name}@{i}"

            # Copy the block to the target device.
            # Same-device copies are handled as local copies by Cluster.
            cluster.copy(
                i,
                j,
                name,
                dst_name=temp_name,
                region=region
            )

            # Record the block name in source-device order.
            received_blocks[j].append(temp_name)

    # Assemble the new shard on each destination device.
    for j in range(num_devices):

        device = cluster.devices[j]
        temp_names = received_blocks[j]

        target_start, target_stop = target_bounds[j]

        if temp_names:

            # Concatenate the received blocks along the ORIGINAL
            # sharding axis, preserving their source-device order.
            device.run(
                name,
                lambda *arrays: np.concatenate(
                    arrays,
                    axis=old_axis
                ),
                *temp_names
            )

            # Release all temporary block buffers after assembly.
            for temp_name in temp_names:
                device.free(temp_name)

        else:

            # This device received no blocks.
            # Create an empty buffer with the correct new shard shape.
            empty_shape = list(shape)
            empty_shape[new_axis] = target_stop - target_start

            device.alloc(
                name,
                np.empty(tuple(empty_shape), dtype=np.float64)
            )

    # Update the existing handle in place.
    t["axis"] = axis
    t["bounds"] = target_bounds

    return t

# Step 7 - map_shards (not yet solved)
# TODO: implement

# Step 8 - reduce_shards
def reduce_shards(cluster, out, t, axis, op="sum"):
    """Reduce the sharded tensor behind t over axis with sum or max; return the result's handle."""

    # Verify that the input tensor is sharded.
    if t["layout"] != "sharded":
        raise ValueError("Input tensor must have a sharded layout")

    # Validate the reduction operation.
    if op not in ("sum", "max"):
        raise ValueError("Operation must be either 'sum' or 'max'")

    num_devices = len(cluster.devices)
    name = t["name"]
    shape = tuple(t["shape"])
    ndim = len(shape)

    # Normalize the reduction axis.
    if not -ndim <= axis < ndim:
        raise ValueError("Reduction axis out of range")

    axis = axis % ndim

    # Normalize the original sharding axis.
    shard_axis = t["axis"] % ndim

    # Obtain the original shard boundaries.
    bounds = [tuple(b) for b in t["bounds"]]

    if len(bounds) != num_devices:
        raise ValueError("Shard bounds must match the number of devices")

    # Calculate the output shape by removing the reduced dimension.
    output_shape = tuple(
        dim for i, dim in enumerate(shape)
        if i != axis
    )

    # Select the appropriate local reduction function.
    if op == "sum":
        def reduce_fn(array):
            return np.sum(array, axis=axis)
    else:
        def reduce_fn(array):
            return np.max(array, axis=axis)

    # ---------------------------------------------------------
    # Case 1: Reduction axis is different from shard axis.
    # ---------------------------------------------------------
    if axis != shard_axis:

        # Each device independently reduces its local shard.
        # The output remains distributed across the same devices.
        for device in cluster.devices:
            device.run(out, reduce_fn, name)

        # Removing a dimension before the sharding axis shifts
        # the sharding axis index down by one.
        new_shard_axis = shard_axis - (1 if axis < shard_axis else 0)

        # Return a handle describing the still-sharded result.
        return handle(
            out,
            output_shape,
            "sharded",
            axis=new_shard_axis,
            bounds=bounds
        )

    # ---------------------------------------------------------
    # Case 2: Reduction axis is the shard axis.
    # Partial results must be combined on device 0.
    # ---------------------------------------------------------

    # Identify devices whose shards contain actual elements
    # along the sharding axis.
    active_devices = [
        i for i, (start, stop) in enumerate(bounds)
        if start < stop
    ]

    # A reduction cannot proceed when every shard is empty.
    if not active_devices:
        raise ValueError("Cannot reduce an entirely empty tensor")

    # Step 1: Calculate one local partial reduction per
    # non-empty device.
    for i in active_devices:
        cluster.devices[i].run(
            out,
            reduce_fn,
            name
        )

    # Step 2: Transfer partial results from nonzero devices
    # to device 0, preserving device order.
    for i in active_devices:

        if i == 0:
            continue

        temp_name = f"{out}@{i}"

        cluster.copy(
            i,
            0,
            out,
            dst_name=temp_name
        )

        # Release the partial result on its original device.
        cluster.devices[i].free(out)

    # Step 3: Prepare the partial results in device order.
    partial_names = [
        out if i == 0 else f"{out}@{i}"
        for i in active_devices
    ]

    # Define the final combine operation.
    # The combination is performed from left to right.
    def combine_partials(*partials):
        result = partials[0]

        for partial in partials[1:]:
            if op == "sum":
                result = result + partial
            else:
                result = np.maximum(result, partial)

        return result

    # Run exactly one combine kernel on device 0.
    # This kernel also runs when there is only one partial result.
    cluster.devices[0].run(
        out,
        combine_partials,
        *partial_names
    )

    # Release all temporary partial-result buffers on device 0.
    for i in active_devices:
        if i != 0:
            cluster.devices[0].free(f"{out}@{i}")

    # Step 4: The final result is a complete tensor on device 0.
    return handle(
        out,
        output_shape,
        "single",
        device=0
    )

# Step 9 - matmul_shards (not yet solved)
# TODO: implement

# Step 10 - place_graph (not yet solved)
# TODO: implement

# Step 11 - allreduce_sum (not yet solved)
# TODO: implement

# Step 12 - train_step (not yet solved)
# TODO: implement

