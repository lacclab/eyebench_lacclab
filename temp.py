import pandas as pd
from pathlib import Path
import pyreadr

read_path = Path('data/MECOL2/stimuli/stimuli.csv')
result = pd.read_csv(read_path)
print(result)
