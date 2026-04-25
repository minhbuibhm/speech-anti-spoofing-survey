import json
import sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'sdd-refactor-v2.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    first_lines = src.split('\n')[:3]
    snippet = ' | '.join(l.strip()[:120] for l in first_lines if l.strip())
    print(f'[{i:03d}] {c["cell_type"]:8s} {len(src):5d}c  {snippet[:160]}')
