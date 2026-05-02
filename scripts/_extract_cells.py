import json, sys, pathlib

for nb_name in ["eval_asvspoof_2019", "eval_asvspoof_2021", "eval_asvspoof_5"]:
    nb_path = pathlib.Path("notebooks") / f"{nb_name}.ipynb"
    nb = json.load(open(nb_path, encoding="utf-8"))
    out_path = pathlib.Path("scripts") / f"_{nb_name}.py"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                f.write(f"# ---- cell {i} ----\n")
                f.writelines(c["source"])
                f.write("\n\n")
    print(nb_name, "->", out_path)
