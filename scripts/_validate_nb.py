"""Validate that every code cell in a notebook compiles as Python."""
import json
import sys
import re

nb_path = sys.argv[1]
nb = json.loads(open(nb_path, encoding="utf-8").read())

errors = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Strip Jupyter magics that aren't valid Python: %pip, !git, etc.
    cleaned_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            cleaned_lines.append(indent + "pass  # magic: " + stripped)
        else:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    try:
        compile(cleaned, f"<cell {i}>", "exec")
        print(f"[{i:03d}] OK   ({len(src)} chars)")
    except SyntaxError as e:
        errors += 1
        print(f"[{i:03d}] FAIL line {e.lineno}: {e.msg}")
        # Show context
        lines = cleaned.splitlines()
        lo = max(0, e.lineno - 3)
        hi = min(len(lines), e.lineno + 2)
        for j in range(lo, hi):
            marker = ">>" if j == e.lineno - 1 else "  "
            print(f"     {marker} {j+1:4d}: {lines[j]}")

print(f"\n{errors} cell(s) failed to compile" if errors else "\nAll code cells compile.")
sys.exit(1 if errors else 0)
