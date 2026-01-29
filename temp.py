import pandas as pd
from pathlib import Path

ia_path = Path('data/OneStopL2/processed/trial_level.feather')
df = pd.read_feather(ia_path)

print(f"Total columns: {len(df.columns)}\n")
print("All columns:")
for i, col in enumerate(sorted(df.columns)[:200], 1):
    print(f"{i:3d}. {col}")

# Inspect tuple-valued columns and any stringified tuple column names
tuple_cols = [c for c in df.columns if isinstance(c, tuple)]
print(f"\nTuple columns ({len(tuple_cols)}):")
for c in tuple_cols:
    print(c)

stringified_tuple_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('(') and c.endswith(')')]
print(f"\nStringified tuple-looking columns ({len(stringified_tuple_cols)}):")
for c in stringified_tuple_cols:
    print(c)
