import pickle
from pathlib import Path

import numpy as np

ROOT = Path("e:/minhbh/HCMUT/projects/252-internship-1/results")

print("=" * 80)
print("ASV19 legacy pickle")
print("=" * 80)
pkl_path = ROOT / "asvspoof19_results" / "all_results_asvspooft_19.pkl"
with open(pkl_path, "rb") as handle:
    payload = pickle.load(handle)

print(f"file: {pkl_path}")
print(f"top-level type: {type(payload).__name__}")
if isinstance(payload, dict):
    print(f"keys: {list(payload.keys())}")
    for model_name, data in payload.items():
        print(f"\n  [{model_name}]")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype} "
                          f"min={v.min():.4f} max={v.max():.4f} "
                          f"first5={v[:5].tolist() if v.ndim == 1 else 'N/A'}")
                elif isinstance(v, (list, tuple)):
                    print(f"    {k}: {type(v).__name__} len={len(v)} first5={list(v[:5])}")
                else:
                    print(f"    {k}: {type(v).__name__} = {v}")
        else:
            print(f"    type: {type(data).__name__}, value: {data}")

print()
print("=" * 80)
print("ASV21 DF npz files")
print("=" * 80)
for npz_name in ["scores_2021DF_AASIST.npz", "scores_2021DF_LFCC_LCNN.npz"]:
    npz_path = ROOT / "asvspoof2021_results" / npz_name
    print(f"\nfile: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    print(f"keys: {list(npz.keys())}")
    for k in npz.keys():
        v = npz[k]
        if v.ndim == 0:
            print(f"  {k}: scalar dtype={v.dtype} value={v.item()}")
        else:
            print(f"  {k}: shape={v.shape} dtype={v.dtype}", end="")
            if v.ndim == 1 and v.size > 0:
                if np.issubdtype(v.dtype, np.number):
                    print(f" min={v.min():.4f} max={v.max():.4f} first5={v[:5].tolist()}")
                else:
                    print(f" first5={v[:5].tolist()}")
            else:
                print()
