import numpy as np

def matmul_tiled(A, B, tile_k=16, tile_n=16):
    """
    Tiled matmul matching the A100 strategy from the HTML:
    - A is M x K, tiled only along K (1-D): each k-slice is M x tile_k
    - B is K x N, tiled 2-D: tiles of tile_k x tile_n
    - C is M x N, tiled along N to match B's column cuts: M x tile_n
    
    Strategy: For each k-slice, load A_tile once, then iterate over n-tiles
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    C = np.zeros((M, N), dtype=A.dtype)
    
    n_k_slices = (K + tile_k - 1) // tile_k
    n_n_tiles = (N + tile_n - 1) // tile_n
    
    for k_slice in range(n_k_slices):
        k_start = k_slice * tile_k
        k_end = min(k_start + tile_k, K)

        A_tile = A[:, k_start:k_end]  # M x tile_k

        for n_tile in range(n_n_tiles):
            n_start = n_tile * tile_n
            n_end = min(n_start + tile_n, N)

            B_tile = B[k_start:k_end, n_start:n_end]  # tile_k x tile_n
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

    print(f"Matrix sizes: A={A.shape}, B={B.shape}")
    print(f"Tiling: tile_k={tile_k}, tile_n={tile_n}")
    print(f"K-slices: {(K + tile_k - 1) // tile_k}, N-tiles: {(N + tile_n - 1) // tile_n}")

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