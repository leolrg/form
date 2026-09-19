# Cached QR tree launch tuning

This is a measured-bottleneck experiment, not an occupancy-based speedup claim.
Frozen V5 leaf/ancestor kernels use 90/92 registers per thread, zero static or
dynamic shared memory, and no reported stack/local spills (`cuobjdump
--dump-resource-usage`). The A100 has 108 SMs. Both kernels factor one node per
warp; a warp-stride loop handles additional nodes assigned to that group.

## Occupancy estimates

For SM80, using a 65,536-register SM, 256-register allocation granularity per
warp, 2,048 threads/64 warps and 32 blocks per SM, both measured register counts
round to 3,072 registers per warp. These are resource estimates; they are not
achieved occupancy measurements.

| Threads/block | Register-limited blocks/SM | Shared-memory limit | Hardware block limit | Thread limit | Estimated resident warps | Estimated occupancy |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 | 21 | None | 32 | 64 | 21 | 32.8% |
| 64 | 10 | None | 32 | 32 | 20 | 31.3% |
| 128 | 5 | None | 32 | 16 | 20 | 31.3% |
| 256 | 2 | None | 32 | 8 | 16 | 25.0% |
| 512 | 1 | None | 32 | 4 | 16 | 25.0% |

The existing QR constructor accepts 32/64/128/256 threads. The 512 row is only a
resource comparison, not a tested or enabled launch. For this experiment keep
32 threads/block: larger blocks do not improve the register occupancy estimate
and reduce the number of SMs available to a small/skewed node set. V5's more
immediate limit is the **grid**, which caps each group at 32 warps. One dominant
group therefore uses at most 32 SMs regardless of spare register capacity.

The cap sweep is 32/128/256 warps per group. For `n` potentially active nodes,
`w = threads/32`, and cap `c`, launch `ceil(min(c,max(1,n))/w)` blocks per group.
Each warp processes a grid-stride sequence of nodes. No work is truncated when
`n` exceeds the cap. Small groups still launch at least one warp and return on
an empty extent; consequently fixed launch overhead can dominate sparse calls.
No `__launch_bounds__` register restriction is proposed: there are no spills to
fix and forcing fewer registers can introduce them.

## Exact extent bound

With first-hole slot allocation, let `h` be the previous highwater extent, `r`
the retained row count after removals, and `c` the current accepted row count.
There are `h-r` holes and `c-r` arrivals. If `c<=h`, all arrivals fit inside the
old extent. Otherwise every old hole fills and exactly `c-h` rows append. Thus

```text
new_highwater = max(old_highwater, accepted_count)
```

The matcher already downloads accepted counts. Mirroring this recurrence, with
all the same invalidation/reset points, gives an exact host launch bound without
an extra device download. A standalone tree API must still support shrinking
extents: it launches through the maximum of the previous and new supplied
bounds so deactivation propagates. Device validation rejects an insufficient
bound through the ordinary root/status download. Layout offsets remain based on
allocated capacity; a smaller launch bound does not reinterpret retained roots.

## Launch and safety checks

The leaf launch has the form:

```cpp
updateTreeLeaves<<<groups * blocks_per_group, threads, 0, stream>>>(...);
```

Ancestor kernels use the same form with their level's bounded node count. The
final partial block has at most `threads/32-1` unused warps; with the chosen
32-thread block there is no partial-warp work unit. Tail groups/levels are checked
against actual device extents. Grid-stride loops cover all nodes, and each QR
always has all 32 lanes participating. With a 256-warp cap and 32-thread blocks,
validate `groups * 256 <= INT_MAX` before launch; the implementation's group
limit must cover the largest enabled cap, not only the original cap of 32.

A runtime occupancy check, if needed to validate the estimates within the CUDA
translation unit, is:

```cpp
int resident_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &resident_blocks, updateTreeLeaves, threads, 0);
```

Compare first/full-dirty, sparse, and unchanged cases separately; a larger grid
can help full refreshes while hurting sparse ones. Select a final configuration
only after clean replay timing and an independent trace. Those results belong
in `certified-rematching-experiments.md`.
