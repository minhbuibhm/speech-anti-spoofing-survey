import pickle
import numpy as np

PKL_PATHS = [
    "results/asvspoof19/results.pkl",
    "results/asvspoof21/results.pkl",
    "results/asvspoof5/results.pkl",
    "results/in_the_wild/results.pkl",
]


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle_compat(path: str):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def inspect_pkl(path: str):
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print("="*60)
    data = load_pickle_compat(path)

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
    for path in PKL_PATHS:
        inspect_pkl(path)
