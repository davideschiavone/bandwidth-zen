import numpy as np

UNITS = 432  # a100's independent tensor cores -- one wave computes this many
             # tiles at once, in parallel, not one at a time


def matmul_tiled(A, B, tile_k=16, tile_n=16, units=UNITS):
    """
    Tiled matmul matching the A100 strategy from the HTML:
    - A is M x K, tiled only along K (1-D): each k-slice is M x tile_k
    - B is K x N, tiled 2-D: tiles of tile_k x tile_n
    - C is M x N, tiled along N to match B's column cuts: M x tile_n

    Tiles are numbered k-major (k-slice outer, n-tile inner) and dispatched
    `units` at a time -- one "wave" per `units` tiles, matching how many
    independent systolic arrays the chip actually has. A wave's tiles can
    span MORE than one k-slice when `units` doesn't divide evenly by the
    number of n-tiles per k-slice -- the common case, not an edge case: with
    the defaults below, n-tiles per k-slice is 125 but units=432, so wave 0
    alone touches k-slices 0, 1, 2 in full and k-slice 3 partially --
    125+125+125+57 = 432 tiles, not 432 tiles of one k-slice.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = np.zeros((M, N), dtype=A.dtype)

    n_k_slices = (K + tile_k - 1) // tile_k
    n_n_tiles = (N + tile_n - 1) // tile_n
    n_tiles_total = n_k_slices * n_n_tiles

    for wave_start in range(0, n_tiles_total, units):
        wave_end = min(wave_start + units, n_tiles_total)
        # Real hardware runs every unit in this wave at once; here we loop
        # one at a time, but each iteration is independent of the others --
        # nothing below depends on another unit's result within the wave.
        for t in range(wave_start, wave_end):
            k_slice, n_tile = divmod(t, n_n_tiles)

            k_start = k_slice * tile_k
            k_end = min(k_start + tile_k, K)
            n_start = n_tile * tile_n
            n_end = min(n_start + tile_n, N)

            A_tile = A[:, k_start:k_end]  # M x tile_k -- shared by every unit
            #                                in this same k-slice
            B_tile = B[k_start:k_end, n_start:n_end]  # tile_k x tile_n -- unique
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
    print(f"Units (tensor cores): {UNITS} -> {n_waves} waves")

    # Every wave's own tile count and k-slice span -- wave 0 was never
    # special, it was just the easiest one to explain. Every interior wave
    # spans several k-slices too, since 432 never divides evenly into
    # 125-tile k-slice rows. The one wave that's genuinely different is the
    # LAST one: unlike every other wave, it doesn't have to be full. Here
    # 23,500 tiles over 432 units leaves the last wave only 172 tiles -- the
    # same "real tile count, not waves * units" idle-slot effect this
    # project's own model corrects for (docs/CORRECTIONS.md D46).
    print()
    for wave_number in range(n_waves):
        wave_start = wave_number * UNITS
        wave_end = min(wave_start + UNITS, n_tiles_total)

        k_slices_seen_in_this_wave = set()
        for t in range(wave_start, wave_end):
            k_slice_of_this_tile = t // n_n_tiles
            k_slices_seen_in_this_wave.add(k_slice_of_this_tile)
        k_slices_seen_in_this_wave = sorted(k_slices_seen_in_this_wave)
        first_k_slice = k_slices_seen_in_this_wave[0]
        last_k_slice = k_slices_seen_in_this_wave[-1]

        tile_count_this_wave = wave_end - wave_start
        fullness = (
            "full"
            if tile_count_this_wave == UNITS
            else f"UNDERFULL, only {tile_count_this_wave} tiles"
        )
        print(
            f"  wave {wave_number:2d}: k-slices {first_k_slice}..{last_k_slice} "
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
