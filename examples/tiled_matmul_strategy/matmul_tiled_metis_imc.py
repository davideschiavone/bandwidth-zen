import numpy as np

# Metis AIPU: a D-IMC (in-memory-compute) chip, not a tensor-core chip like
# the A100 script (matmul_tiled.py) models. The array itself IS the weight
# storage -- a B-tile has to be WRITTEN into one of the array's resident
# "weight sets" before any compute can use it. A tensor core has no such
# step: it reads both operands fresh, every instruction, and stores nothing
# (docs/CORRECTIONS.md D30/D33). That write is the whole strategy difference.
UNITS = 4          # Metis's 4 independent AI-core arrays
WEIGHT_SETS = 4    # resident B-tile slots PER array -- 16 chip-wide
TILE_K = 512       # array rows -> K per tile (512x512, not the A100's 16x16:
TILE_N = 512       #   far fewer arrays, far bigger tiles)


def matmul_tiled_imc(A, B, tile_k=TILE_K, tile_n=TILE_N, units=UNITS, weight_sets=WEIGHT_SETS):
    """
    Tiled matmul matching the Metis AIPU strategy:
    - A is M x K, tiled only along K (1-D) -- same convention as the A100
      script: A crosses DRAM once per k-slice, never tiled along M.
    - B is K x N, tiled 2-D into (tile_k x tile_n) blocks, same as the A100
      script -- but the blocks themselves are 512x512, not 16x16.
    - Before a tile can be multiplied, it must be WRITTEN into one of this
      array's resident weight sets (imc_write). Wave `w`'s tiles all land in
      weight-set index `w % weight_sets`, reusing (overwriting) whatever
      tile occupied that slot before. With units=4 and weight_sets=4, 16
      tiles fit resident at once; a shape needing more than 16 tiles forces
      some of them to displace an earlier one -- real hardware cost this
      script counts, that a tensor-core script has no equivalent of at all.
    - C is M x N, accumulated across k-slices exactly like the A100 script.

    Returns (C, rewrites): rewrites is how many tiles found their assigned
    weight-set slot already occupied by a DIFFERENT tile.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = np.zeros((M, N), dtype=A.dtype)

    n_k_slices = (K + tile_k - 1) // tile_k
    n_n_tiles = (N + tile_n - 1) // tile_n
    n_tiles_total = n_k_slices * n_n_tiles
    n_waves = (n_tiles_total + units - 1) // units

    # resident[unit][weight_set] = which global tile index currently
    # occupies that slot, or None if the slot has never been written.
    resident = [[None] * weight_sets for _ in range(units)]
    rewrites = 0

    for wave_number in range(n_waves):
        weight_set_index = wave_number % weight_sets
        wave_start = wave_number * units
        wave_end = min(wave_start + units, n_tiles_total)

        for unit, t in enumerate(range(wave_start, wave_end)):
            # imc_write: this tile must be resident before it can compute.
            if resident[unit][weight_set_index] is not None:
                rewrites += 1  # this slot already held a DIFFERENT tile
            resident[unit][weight_set_index] = t

            k_slice, n_tile = divmod(t, n_n_tiles)
            k_start = k_slice * tile_k
            k_end = min(k_start + tile_k, K)
            n_start = n_tile * tile_n
            n_end = min(n_start + tile_n, N)

            A_tile = A[:, k_start:k_end]  # M x tile_k -- shared by every unit
            #                                in this same k-slice
            B_tile = B[k_start:k_end, n_start:n_end]  # now resident in the array
            C_tile = C[:, n_start:n_end]

            C_tile += A_tile @ B_tile

    return C, rewrites


def main():
    M, K, N = 1000, 3000, 2000

    np.random.seed(42)
    # Same reasoning as the A100 script: integers give an exact match, no
    # float-tolerance to reason about.
    A = np.random.randint(-10, 11, size=(M, K)).astype(np.int64)
    B = np.random.randint(-10, 11, size=(K, N)).astype(np.int64)

    n_k_slices = (K + TILE_K - 1) // TILE_K
    n_n_tiles = (N + TILE_N - 1) // TILE_N
    n_tiles_total = n_k_slices * n_n_tiles
    n_waves = (n_tiles_total + UNITS - 1) // UNITS
    resident_capacity = UNITS * WEIGHT_SETS

    print(f"Matrix sizes: A={A.shape}, B={B.shape}")
    print(f"Tiling: tile_k={TILE_K}, tile_n={TILE_N} (Metis's 512x512 array, not A100's 16x16)")
    print(f"K-slices: {n_k_slices}, N-tiles: {n_n_tiles}, total tiles: {n_tiles_total}")
    print(f"Units (AI cores): {UNITS} -> {n_waves} waves")
    print(
        f"Weight sets per unit: {WEIGHT_SETS} -> {resident_capacity} tiles resident "
        f"chip-wide at once"
    )

    C_tiled, rewrites = matmul_tiled_imc(A, B)
    C_golden = A @ B

    expected_rewrites = max(0, n_tiles_total - resident_capacity)
    print(
        f"\nWeight-set rewrites: {rewrites} "
        f"(expected {n_tiles_total} tiles - {resident_capacity} resident slots "
        f"= {expected_rewrites})"
    )

    max_abs_diff = np.max(np.abs(C_tiled - C_golden))
    print(f"Max absolute difference: {max_abs_diff}")

    if np.array_equal(C_tiled, C_golden) and rewrites == expected_rewrites:
        print("✓ PASSED: Tiled matmul matches golden model exactly, rewrite count checks out")
    else:
        print("✗ FAILED")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
