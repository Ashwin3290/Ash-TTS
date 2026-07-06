import json
from pathlib import Path

with open('data/processed/train_manifest.json') as f:
    manifest = json.load(f)

pd = Path('data/processed')
missing = [item['id'] for item in manifest if not (pd / 'duration' / f'{item["id"]}.npy').exists()]
print(f'Missing duration files: {len(missing)} / {len(manifest)}')
print('First 5:', missing[:5])