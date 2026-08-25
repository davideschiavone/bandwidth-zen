import numpy as np

# Two different things, two different words -- keeping them straight is the
# whole point of the naming below:
#   CORE  -- one of the chip's independent tensor cores. There are UNITS of
#            them and they all compute AT THE SAME INSTANT, in parallel.
#            `core_id` indexes this, 0 .. UNITS-1.
#   WAVE  -- one whole-chip iteration: every core busy once. `wave` indexes
#            this, 0 .. n_waves-1. It is the SEQUENTIAL dimension -- wave w+1
#            cannot start until the chip has finished wave w.
# So the 55 in this example is a wave count (how many times the whole chip
# has to go round), never a core count.
UNITS = 432  # a100's independent tensor cores -- how many run in parallel


def matmul_tiled(A, B, tile_k=16, tile_n=16, units=UNITS):
    """
    Tiled matmul matching the A100 strategy from the HTML:
    - A is M x K, tiled only along K (1-D): each k-slice is M x tile_k
    - B is K x N, tiled 2-D: tiles of tile_k x tile_n
    - C is M x N, tiled along N to match B's column cuts: M x tile_n

    Tiles are numbered k-major (k-slice outer, n-tile inner). Tile
    `wave * units + core_id` is the one core `core_id` handles during wave
    `wave` -- so the inner loop below is the PARALLEL dimension (all cores at
    the same instant) and the outer loop is the SEQUENTIAL one (one whole-chip
    iteration after another).

    One wave's tiles can span MORE than one k-slice, whenever `units` doesn't
    divide evenly by the number of n-tiles per k-slice -- the common case, not
    an edge case: with the defaults below, n-tiles per k-slice is 125 but
    units=432, so wave 0 alone touches k-slices 0, 1, 2 in full and k-slice 3
    partially -- 125+125+125+57 = 432 tiles, not 432 tiles of one k-slice.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = np.zeros((M, N), dtype=A.dtype)

    n_k_slices = (K + tile_k - 1) // tile_k
    n_n_tiles = (N + tile_n - 1) // tile_n
    n_tiles_total = n_k_slices * n_n_tiles
    n_waves = (n_tiles_total + units - 1) // units

    for wave in range(n_waves):  # SEQUENTIAL: one whole-chip iteration each
        first_tile_in_wave = wave * units
        print(f"wave {wave}: tiles {first_tile_in_wave}..{min(first_tile_in_wave + units, n_tiles_total) - 1}")

        # PARALLEL: on real hardware every core_id below runs at the same
        # instant. The loop is sequential only because Python is -- nothing
        # in an iteration depends on another core's result within this wave.
        for core_id in range(units):
            tile_index = first_tile_in_wave + core_id
            if tile_index >= n_tiles_total:
                break  # last wave is underfull: these cores sit idle (D46)

            k_slice, n_tile = divmod(tile_index, n_n_tiles)
            k_start = k_slice * tile_k
            k_end = min(k_start + tile_k, K)
            if wave in (0, 1) and core_id == 0:
                print(f"  wave={wave} core_id={core_id} tile_index={tile_index}: "
                      f"k_slice={k_slice}, n_tile={n_tile}, k_start={k_start}, k_end={k_end}")

            n_start = n_tile * tile_n
            n_end = min(n_start + tile_n, N)

            A_tile = A[:, k_start:k_end]  # M x tile_k -- shared by every core
            #                                working on this same k-slice
            B_tile = B[k_start:k_end, n_start:n_end]  # tile_k x tile_n -- this
            #                                            core's own tile
            C_tile = C[:, n_start:n_end]  # M x tile_n

            C_tile += A_tile @ B_tile

    return C


def main():
    M, K, N = 1000, 3000, 2000
    tile_k, tile_n = 16, 16

    np.random.seed(42)
    # Integers, not float32: exact arithmetic means the tiled and golden sums
    # must match bit-for-bit, regardless of accumulation order -- no rtol/atol
    # tolerance to reason about. int64 accumulator: worst case a term is
    # 10*10=100, summed over K=3000 terms is <=300,000, nowhere near overflow.
    A = np.random.randint(-10, 11, size=(M, K)).astype(np.int64)
    B = np.random.randint(-10, 11, size=(K, N)).astype(np.int64)

    n_k_slices = (K + tile_k - 1) // tile_k
    n_n_tiles = (N + tile_n - 1) // tile_n
    n_tiles_total = n_k_slices * n_n_tiles
    n_waves = (n_tiles_total + UNITS - 1) // UNITS

    print(f"Matrix sizes: A={A.shape}, B={B.shape}")
    print(f"Tiling: tile_k={tile_k}, tile_n={tile_n}")
    print(f"K-slices: {n_k_slices}, N-tiles: {n_n_tiles}, total tiles: {n_tiles_total}")
    print(f"Cores (UNITS, run in parallel): {UNITS}")
    print(f"Waves (whole-chip iterations, run in sequence): {n_waves}")

    # Every wave's own core usage and k-slice span -- wave 0 was never
    # special, it was just the easiest one to explain. Every interior wave
    # spans several k-slices too, since 432 never divides evenly into
    # 125-tile k-slice rows. The one wave that's genuinely different is the
    # LAST one: unlike every other wave, it doesn't have to keep every core
    # busy. Here 23,500 tiles over 432 cores leaves the last wave only 172
    # busy cores -- the same "real tile count, not waves * units" idle-slot
    # effect this project's own model corrects for (docs/CORRECTIONS.md D46).
    print()
    for wave in range(n_waves):
        first_tile_in_wave = wave * UNITS
        end_tile_in_wave = min(first_tile_in_wave + UNITS, n_tiles_total)

        k_slices_seen_in_this_wave = set()
        for tile_index in range(first_tile_in_wave, end_tile_in_wave):
            k_slice_of_this_tile = tile_index // n_n_tiles
            k_slices_seen_in_this_wave.add(k_slice_of_this_tile)
        k_slices_seen_in_this_wave = sorted(k_slices_seen_in_this_wave)
        first_k_slice = k_slices_seen_in_this_wave[0]
        last_k_slice = k_slices_seen_in_this_wave[-1]

        busy_cores = end_tile_in_wave - first_tile_in_wave
        fullness = (
            "all cores busy"
            if busy_cores == UNITS
            else f"UNDERFULL, only {busy_cores}/{UNITS} cores busy"
        )
        print(
            f"  wave {wave:2d}: k-slices {first_k_slice}..{last_k_slice} "
            f"({len(k_slices_seen_in_this_wave)} of them) -- {fullness}"
        )

    C_tiled = matmul_tiled(A, B, tile_k, tile_n)
    C_golden = A @ B

    max_abs_diff = np.max(np.abs(C_tiled - C_golden))
    print(f"\nMax absolute difference: {max_abs_diff}")

    if np.array_equal(C_tiled, C_golden):
        print("✓ PASSED: Tiled matmul matches golden model exactly")
    else:
        print("✗ FAILED: Results differ")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
