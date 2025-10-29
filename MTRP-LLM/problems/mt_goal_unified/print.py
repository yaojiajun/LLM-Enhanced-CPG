import os
import numpy as np

def peek_npz(path: str, idx: int = 0):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return

    with np.load(path, allow_pickle=True) as f:
        print(f"=== NPZ file: {path} ===")
        print("Keys:", f.files)

        # 打印每个键的 shape/dtype
        print("\n--- Array summary (shape / dtype) ---")
        for k in f.files:
            arr = f[k]
            print(f"{k:>20}: shape={arr.shape}, dtype={arr.dtype}")

        # 如果有 node_coords，顺便打印 N（含两端仓库）
        if "node_coords" in f.files:
            B, N, _ = f["node_coords"].shape
            print(f"\nBatch size (B) = {B}, Nodes per instance (N) = {N}")

        # 打印第 idx 条样本的预览
        print(f"\n=== Sample [{idx}] preview ===")
        for k in f.files:
            arr = f[k]
            if arr.ndim == 0:
                # 标量
                print(f"{k}[{idx}]: {arr}")
                continue
            if arr.shape[0] <= idx:
                print(f"{k}[{idx}]: (index out of range for first dim {arr.shape[0]})")
                continue

            s = arr[idx]  # 取第 idx 条样本
            if s.ndim == 0:
                print(f"{k}[{idx}]: {s}")
            elif s.ndim == 1:
                head = np.array2string(s[:10], precision=4, separator=", ")
                print(f"{k}[{idx}] shape={s.shape}, head={head}")
            elif s.ndim == 2:
                rows = min(5, s.shape[0])
                print(f"{k}[{idx}] shape={s.shape}, first {rows} rows:\n{s[:rows]}")
            elif s.ndim == 3:
                rows = min(3, s.shape[0])
                print(f"{k}[{idx}] shape={s.shape}, first {rows} rows (full cols):\n{s[:rows]}")
            else:
                print(f"{k}[{idx}] (ndim={s.ndim}) -> showing shape only: {s.shape}")

if __name__ == "__main__":
    peek_npz("data/VRPTW/vrptw100_uniform.pkl", idx=0)
    # peek_npz("data/VRPTW/cvrp100_test.npz", idx=0)