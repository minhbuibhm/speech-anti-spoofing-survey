import json
import sys

nb_path = sys.argv[1]
indices = [int(x) for x in sys.argv[2].split(',')]
nb = json.load(open(nb_path, encoding='utf-8'))
cells = nb['cells']
for i in indices:
    c = cells[i]
    print(f'=== CELL [{i}] type={c["cell_type"]} ===')
    print(''.join(c['source']))
    print()
