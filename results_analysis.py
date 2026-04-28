import pickle
import numpy as np

PKL_19 = "results/asvspoof19/results.pkl"
PKL_21 = "results/asvspoof21/results.pkl"


def inspect_pkl(path: str):
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print("="*60)
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys ({len(data)}): {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"  [{k}] ndarray  shape={v.shape}  dtype={v.dtype}  sample={v[:3]}")
            elif isinstance(v, list):
                print(f"  [{k}] list  len={len(v)}  first={v[0] if v else 'empty'}")
            else:
                print(f"  [{k}] {type(v).__name__}  value={v}")
    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        print(f"First element type: {type(data[0])}")
        print(f"First element: {data[0]}")
    elif isinstance(data, np.ndarray):
        print(f"ndarray  shape={data.shape}  dtype={data.dtype}")
        print(f"Sample (first 5): {data[:5]}")
    else:
        print(f"Value: {data}")


if __name__ == "__main__":
    inspect_pkl(PKL_19)
    inspect_pkl(PKL_21)
