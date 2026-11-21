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

# Step 7 - map_shards
def map_shards(cluster, out, fn, *tensors):
    """Run the elementwise kernel fn on every device and return the handle of the result."""

    # At least one operand is required.
    if not tensors:
        raise ValueError("At least one tensor operand is required")

    # Validate all operand layouts before performing any copies or kernels.
    for tensor in tensors:
        if tensor["layout"] not in ("sharded", "replicated"):
            raise ValueError(
                "Single-layout operands must be scattered or replicated first"
            )

    # Identify all sharded operands.
    sharded_tensors = [
        tensor for tensor in tensors
        if tensor["layout"] == "sharded"
    ]

    # ---------------------------------------------------------
    # Case 1: At least one operand is sharded.
    # ---------------------------------------------------------
    if sharded_tensors:

        # The first sharded operand is the reference layout.
        reference = sharded_tensors[0]

        reference_shape = tuple(reference["shape"])
        reference_axis = reference["axis"]
        reference_bounds = reference["bounds"]

        # Validate the shapes of all sharded operands before
        # performing any re-sharding or kernel execution.
        for tensor in sharded_tensors[1:]:

            if tuple(tensor["shape"]) != reference_shape:
                raise ValueError(
                    "All sharded operands must have the same shape"
                )

        # Align every other sharded operand with the reference.
        # reshard() updates the corresponding handle in place.
        for tensor in sharded_tensors[1:]:

            reshard(
                cluster,
                tensor,
                reference_axis,
                bounds=reference_bounds
            )

        # Every device executes the kernel using its own local
        # buffers. Replicated operands remain whole, allowing
        # normal NumPy broadcasting against the local shards.
        for device in cluster.devices:

            input_names = [
                tensor["name"]
                for tensor in tensors
            ]

            device.run(
                out,
                fn,
                *input_names
            )

        # Return a handle describing the sharded output.
        return handle(
            out,
            reference_shape,
            "sharded",
            axis=reference_axis,
            bounds=reference_bounds
        )

    # ---------------------------------------------------------
    # Case 2: All operands are replicated.
    # ---------------------------------------------------------

    # Every device executes the kernel using its own replicas.
    for device in cluster.devices:

        input_names = [
            tensor["name"]
            for tensor in tensors
        ]

        device.run(
            out,
            fn,
            *input_names
        )

    # Determine the output shape from the produced buffer.
    output_shape = tuple(cluster.devices[0].buffers[out].shape)

    # Return a handle describing the replicated output.
    return handle(
        out,
        output_shape,
        "replicated"
    )

# Step 8 - reduce_shards
def reduce_shards(cluster, out, t, axis, op="sum"):
    """Reduce the sharded tensor behind t over axis with sum or max; return the result's handle."""

    # Validate the tensor layout.
    if t["layout"] != "sharded":
        raise ValueError("Input tensor must have sharded layout")

    # Validate the reduction operation.
    if op not in ("sum", "max"):
        raise ValueError("Operation must be 'sum' or 'max'")

    shape = tuple(t["shape"])
    ndim = len(shape)
    num_devices = len(cluster.devices)

    if ndim == 0:
        raise ValueError("Cannot reduce a scalar tensor along an axis")

    # Normalize the reduction and sharding axes.
    reduce_axis = axis % ndim
    shard_axis = t["axis"] % ndim

    # The reduced axis is removed from the output shape.
    output_shape = tuple(
        dim for i, dim in enumerate(shape)
        if i != reduce_axis
    )

    # Select the appropriate local reduction operation.
    reducer = np.sum if op == "sum" else np.max

    # ---------------------------------------------------------
    # Case 1: Reduction axis is NOT the sharding axis.
    # ---------------------------------------------------------
    if reduce_axis != shard_axis:

        # Each device independently reduces its local shard.
        # Every device executes exactly one kernel.
        for device in cluster.devices:

            device.run(
                out,
                lambda x: reducer(x, axis=reduce_axis),
                t["name"]
            )

        # Adjust the sharding axis because the reduced dimension
        # has been removed from the output shape.
        new_shard_axis = (
            shard_axis - 1
            if reduce_axis < shard_axis
            else shard_axis
        )

        # The output remains sharded with the same boundaries.
        return handle(
            out,
            output_shape,
            "sharded",
            axis=new_shard_axis,
            bounds=t["bounds"]
        )

    # ---------------------------------------------------------
    # Case 2: Reduction axis IS the sharding axis.
    # ---------------------------------------------------------

    # Identify devices with non-empty shards.
    bounds = t["bounds"]

    if len(bounds) != num_devices:
        raise ValueError("Shard bounds must match the number of devices")

    nonempty_devices = [
        i for i, (start, stop) in enumerate(bounds)
        if start < stop
    ]

    # A distributed reduction cannot proceed if no device
    # contains any elements along the sharding axis.
    if not nonempty_devices:
        raise ValueError("Cannot reduce when every shard is empty")

    # Step 1: Compute local partial reductions.
    # Empty shards do not execute kernels.
    for i in nonempty_devices:

        cluster.devices[i].run(
            out,
            lambda x: reducer(x, axis=reduce_axis),
            t["name"]
        )

    # Step 2: Transfer remote partial results to device 0.
    # Store each partial under a temporary name.
    # Free the remote partial after a successful copy.
    for i in nonempty_devices:

        if i == 0:
            continue

        temp_name = f"{out}@{i}"

        cluster.copy(
            i,
            0,
            out,
            dst_name=temp_name
        )

        cluster.devices[i].free(out)

    # Step 3: Prepare partial results in device order.
    partial_names = [
        out if i == 0 else f"{out}@{i}"
        for i in nonempty_devices
    ]

    # Define the combine operation.
    # Partials are folded from left to right.
    def combine_partials(*partials):

        result = partials[0]

        for partial in partials[1:]:

            if op == "sum":
                result = result + partial

            else:
                result = np.maximum(result, partial)

        return result

    # Device 0 executes exactly one combine kernel.
    # This also runs when only one partial result exists.
    cluster.devices[0].run(
        out,
        combine_partials,
        *partial_names
    )

    # Step 4: Free all temporary partial buffers.
    for i in nonempty_devices:

        if i != 0:
            cluster.devices[0].free(f"{out}@{i}")

    # The final result is a complete tensor on device 0.
    return handle(
        out,
        output_shape,
        "single",
        device=0
    )

# Step 9 - matmul_shards
def matmul_shards(cluster, out, x, w):
    """Multiply row-sharded x by replicated w on every device; return the result's handle."""

    # Validate that X is sharded.
    if x["layout"] != "sharded":
        raise ValueError("X must have sharded layout")

    # Validate the matrix dimensions.
    x_shape = tuple(x["shape"])
    w_shape = tuple(w["shape"])

    if len(x_shape) != 2 or len(w_shape) != 2:
        raise ValueError("X and W must both be two-dimensional tensors")

    n, d = x_shape
    w_d, m = w_shape

    # The inner dimensions of the matrices must match.
    if d != w_d:
        raise ValueError(
            f"Incompatible matrix dimensions: X has {d} columns "
            f"but W has {w_d} rows"
        )

    # Validate the layout of W before performing any operations.
    if w["layout"] == "sharded":
        raise ValueError("W must be replicated or single, not sharded")

    if w["layout"] not in ("replicated", "single"):
        raise ValueError("Unsupported layout for W")

    num_devices = len(cluster.devices)

    # Validate the sharding axis of X.
    if x["axis"] not in (-2, -1, 0, 1):
        raise ValueError("X must be sharded along axis 0 or axis 1")

    # Normalize the axis to support negative axis indices.
    x_axis = x["axis"] % 2

    # Validate the source device when W is a single-device tensor.
    if w["layout"] == "single":
        home = w["device"]

        if (
            not isinstance(home, (int, np.integer))
            or not -num_devices <= home < num_devices
        ):
            raise ValueError("W has an invalid source device")

    # ---------------------------------------------------------
    # Step 1: Ensure X is sharded along axis 0 (rows).
    # ---------------------------------------------------------
    if x_axis == 1:
        reshard(cluster, x, axis=0)

    # ---------------------------------------------------------
    # Step 2: Ensure W is replicated across all devices.
    # ---------------------------------------------------------
    if w["layout"] == "single":

        # Replicate W from its current device.
        replicate(
            cluster,
            w["name"],
            home=w["device"]
        )

        # Update the existing W handle in place.
        w.update({
            "layout": "replicated",
            "axis": None,
            "bounds": None,
            "device": None
        })

    # ---------------------------------------------------------
    # Step 3: Perform local matrix multiplication on each device.
    # ---------------------------------------------------------
    for device in cluster.devices:

        # Each device multiplies its local shard of X
        # by its complete local replica of W.
        device.run(
            out,
            lambda a, b: a @ b,
            x["name"],
            w["name"]
        )

    # ---------------------------------------------------------
    # Step 4: Return the row-sharded output handle.
    # ---------------------------------------------------------
    return handle(
        out,
        (n, m),
        "sharded",
        axis=0,
        bounds=x["bounds"]
    )

# Step 10 - place_graph
def place_graph(kernels, inputs, n_devices, mem_limit):
    """Place every kernel on a device with the fewest bytes copied under the memory limit."""

    # No valid placement is possible without any devices.
    if n_devices <= 0:
        return None

    n_kernels = len(kernels)

    best_result = None
    best_assignment = None

    # ---------------------------------------------------------
    # Step 1: Enumerate every possible kernel-to-device assignment.
    # ---------------------------------------------------------
    # Each kernel can execute on any of the n_devices devices.
    # The number of possible placements is n_devices ** n_kernels.
    #
    # Assignments are generated in lexicographic order so that
    # ties can be resolved deterministically.

    for code in range(n_devices ** n_kernels):

        # Decode the integer into a device assignment tuple.
        assignment = [0] * n_kernels
        remaining = code

        for i in range(n_kernels - 1, -1, -1):
            assignment[i] = remaining % n_devices
            remaining //= n_devices

        assignment = tuple(assignment)

        # -----------------------------------------------------
        # Step 2: Initialize memory, buffer homes and sizes.
        # -----------------------------------------------------

        # Every device initially holds its assigned input buffers.
        memory = [0] * n_devices

        # Track the original home device of every buffer.
        buffer_home = {}

        # Track the size of every buffer in bytes.
        buffer_size = {}

        # Track which buffers are present on each device.
        resident = [set() for _ in range(n_devices)]

        # Account for all graph input buffers.
        for name, info in inputs.items():

            device = info["device"]
            nbytes = info["bytes"]

            buffer_home[name] = device
            buffer_size[name] = nbytes

            memory[device] += nbytes
            resident[device].add(name)

        # Reject assignments whose initial inputs already
        # exceed the memory limit.
        if any(used > mem_limit for used in memory):
            continue

        bytes_moved = 0
        schedule = []
        valid = True

        # -----------------------------------------------------
        # Step 3: Simulate kernels in dependency order.
        # -----------------------------------------------------

        for kernel_index, kernel in enumerate(kernels):

            kernel_name = kernel["name"]
            kernel_inputs = kernel["inputs"]
            kernel_bytes = kernel["bytes"]

            # Select the assigned device for this kernel.
            device = assignment[kernel_index]

            # -------------------------------------------------
            # Step 4: Ensure all input buffers are available.
            # -------------------------------------------------

            # Process inputs in their original argument order.
            for buffer in kernel_inputs:

                # Find the buffer's original home device.
                source = buffer_home[buffer]

                # Copy only if the required buffer is not already
                # present on the assigned execution device.
                if buffer not in resident[device]:

                    nbytes = buffer_size[buffer]

                    # The copy always originates from the buffer's
                    # original home device.
                    schedule.append(
                        ("copy", buffer, source, device)
                    )

                    # Count communication and destination memory.
                    bytes_moved += nbytes
                    memory[device] += nbytes

                    # Record the buffer as resident on this device.
                    # This prevents repeated copies for later kernels.
                    resident[device].add(buffer)

                    # Reject the placement if memory is exceeded.
                    if memory[device] > mem_limit:
                        valid = False
                        break

            if not valid:
                break

            # -------------------------------------------------
            # Step 5: Execute the kernel on its assigned device.
            # -------------------------------------------------

            schedule.append(
                ("run", kernel_name, device)
            )

            # The kernel's output is stored permanently on its
            # execution device.
            memory[device] += kernel_bytes

            # Register the output buffer and its home device.
            buffer_home[kernel_name] = device
            buffer_size[kernel_name] = kernel_bytes
            resident[device].add(kernel_name)

            # Every output remains allocated, so check memory.
            if memory[device] > mem_limit:
                valid = False
                break

        # -----------------------------------------------------
        # Step 6: Discard invalid placements.
        # -----------------------------------------------------

        if not valid:
            continue

        # -----------------------------------------------------
        # Step 7: Keep the placement with the lowest transfer cost.
        # -----------------------------------------------------

        # Assignments are generated lexicographically.
        # Therefore, retaining the first placement on equal costs
        # automatically satisfies the required tie-breaking rule.
        if (
            best_result is None
            or bytes_moved < best_result["bytes_moved"]
        ):

            best_assignment = assignment

            best_result = {
                "assignment": {
                    kernels[i]["name"]: assignment[i]
                    for i in range(n_kernels)
                },
                "bytes_moved": bytes_moved,
                "memory": memory,
                "schedule": schedule
            }

    # Return None if every placement violates a memory limit.
    if best_result is None:
        return None

    return best_result

# Step 11 - allreduce_sum
def allreduce_sum(cluster, name, out=None):
    """Leave the sum of every device's buffer on every device; return the bytes moved."""

    # Use the original buffer name when no output name is specified.
    if out is None:
        out = name

    num_devices = len(cluster.devices)

    if num_devices == 0:
        raise ValueError("All-reduce requires at least one device")

    # Record the communication counter before starting.
    moved_before = cluster.bytes_moved

    # ---------------------------------------------------------
    # Step 1: Copy gradients from all devices to device 0.
    # ---------------------------------------------------------

    # Preserve the source gradient on each device.
    # Store remote gradients on device 0 under temporary names.
    temp_names = []

    # All buffers must have identical shapes.
    reference_shape = cluster.devices[0].buffers[name].shape

    for i in range(1, num_devices):

        source = cluster.devices[i].buffers[name]

        if source.shape != reference_shape:
            raise ValueError("All gradient buffers must have the same shape")

    # Copy remote gradients to device 0 in device order.
    for i in range(1, num_devices):

        temp_name = f"{name}@{i}"

        cluster.copy(
            i,
            0,
            name,
            dst_name=temp_name
        )

        temp_names.append(temp_name)

    # ---------------------------------------------------------
    # Step 2: Sum all gradients on device 0.
    # ---------------------------------------------------------

    def combine_gradients(*gradients):
        """Add gradient arrays using a left-to-right accumulation."""

        result = gradients[0]

        for gradient in gradients[1:]:
            result = result + gradient

        return result

    # The original gradient on device 0 comes first,
    # followed by the temporary buffers in device order.
    input_names = [name] + temp_names

    # Execute exactly one combine kernel, even for one device.
    cluster.devices[0].run(
        out,
        combine_gradients,
        *input_names
    )

    # ---------------------------------------------------------
    # Step 3: Free the temporary gradient buffers.
    # ---------------------------------------------------------

    # Preserve the output if its name happens to coincide
    # with one of the temporary buffer names.
    for temp_name in temp_names:

        if temp_name != out:
            cluster.devices[0].free(temp_name)

    # ---------------------------------------------------------
    # Step 4: Broadcast the summed gradient to every device.
    # ---------------------------------------------------------

    for i in range(1, num_devices):

        cluster.copy(
            0,
            i,
            out
        )

    # Return the total communication cost of this operation.
    return cluster.bytes_moved - moved_before

# Step 12 - train_step
def train_step(cluster, x, y, w, lr):
    """One data-parallel gradient step for the linear model; returns the loss before the update."""

    # Extract the matrix dimensions.
    n, d = tuple(x["shape"])
    y_n, m = tuple(y["shape"])
    w_d, w_m = tuple(w["shape"])

    # ---------------------------------------------------------
    # Step 1: Validate the input tensors.
    # ---------------------------------------------------------

    if x["layout"] != "sharded" or y["layout"] != "sharded":
        raise ValueError("X and Y must have sharded layout")

    if w["layout"] != "replicated":
        raise ValueError("W must have replicated layout")

    if len(x["shape"]) != 2 or len(y["shape"]) != 2 or len(w["shape"]) != 2:
        raise ValueError("X, Y, and W must be two-dimensional tensors")

    if x["axis"] not in (0, -2) or y["axis"] not in (0, -2):
        raise ValueError("X and Y must be sharded along axis 0")

    if y_n != n or w_d != d or w_m != m:
        raise ValueError("Incompatible shapes for X, Y, and W")

    # Check that X and Y use identical shard boundaries.
    x_bounds = [tuple(b) for b in x["bounds"]]
    y_bounds = [tuple(b) for b in y["bounds"]]

    if x_bounds != y_bounds:
        raise ValueError("X and Y must have the same shard boundaries")

    if len(x_bounds) != len(cluster.devices):
        raise ValueError("Shard count must match the number of devices")

    if n <= 0:
        raise ValueError("Training requires a non-empty batch")

    # Names of the six temporary buffers used during training.
    temporary_names = (
        "_pred",
        "_res",
        "_grad",
        "_sq",
        "_rowsq",
        "_loss"
    )

    try:

        # -----------------------------------------------------
        # Step 2: Calculate predictions: _pred = X @ W
        # -----------------------------------------------------

        # Multiply each device's local rows of X by its replica of W.
        pred = matmul_shards(
            cluster,
            "_pred",
            x,
            w
        )

        # -----------------------------------------------------
        # Step 3: Calculate residuals: _res = _pred - Y
        # -----------------------------------------------------

        residual = map_shards(
            cluster,
            "_res",
            lambda prediction, target: prediction - target,
            pred,
            y
        )

        # -----------------------------------------------------
        # Step 4: Calculate local gradients: _grad = X.T @ R
        # -----------------------------------------------------

        # Every device calculates a partial gradient using its
        # own X shard and corresponding residual shard.
        for device in cluster.devices:

            device.run(
                "_grad",
                lambda x_shard, residual_shard: (
                    x_shard.T @ residual_shard
                ),
                x["name"],
                residual["name"]
            )

        # -----------------------------------------------------
        # Step 5: Aggregate gradients using All-Reduce.
        # -----------------------------------------------------

        # Sum all local gradients and distribute the result
        # back to every device.
        allreduce_sum(
            cluster,
            "_grad"
        )

        # -----------------------------------------------------
        # Step 6: Update weights on every device.
        # -----------------------------------------------------

        # Apply the same gradient update to every replica of W.
        # The division by n converts the summed gradient into
        # the average gradient over the training batch.
        for device in cluster.devices:

            device.run(
                w["name"],
                lambda weights, gradient: (
                    weights - (lr / n) * gradient
                ),
                w["name"],
                "_grad"
            )

        # -----------------------------------------------------
        # Step 7: Calculate squared residuals: _sq = R * R
        # -----------------------------------------------------

        squared = map_shards(
            cluster,
            "_sq",
            lambda r: r * r,
            residual
        )

        # -----------------------------------------------------
        # Step 8: Sum squared residuals along axis 1.
        # _rowsq = sum(_sq, axis=1)
        # -----------------------------------------------------

        row_sums = reduce_shards(
            cluster,
            "_rowsq",
            squared,
            axis=1,
            op="sum"
        )

        # -----------------------------------------------------
        # Step 9: Sum row-wise results along axis 0.
        # _loss = sum(_rowsq, axis=0)
        # -----------------------------------------------------

        total_loss = reduce_shards(
            cluster,
            "_loss",
            row_sums,
            axis=0,
            op="sum"
        )

        # Read the complete loss from device 0.
        # Divide by 2n to obtain the specified mean squared
        # error objective with the factor of one-half.
        loss = (
            float(cluster.devices[0].buffers["_loss"].item())
            / (2 * n)
        )

        # -----------------------------------------------------
        # Step 10: Return the loss before the weight update.
        # -----------------------------------------------------

        return float(loss)

    finally:

        # -----------------------------------------------------
        # Step 11: Free all six temporary buffers.
        # -----------------------------------------------------

        # Remove temporary buffers from every device that holds them.
        # Preserve the original inputs and updated weight replicas.
        for device in cluster.devices:

            for name in temporary_names:

                if name in device.buffers:
                    device.free(name)

