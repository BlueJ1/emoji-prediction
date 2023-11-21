import pandas as pd
from pathlib import Path

from parse_to_df import parse_to_df

data_path = Path(__file__).parent.parent / 'data'
file_name = 'train.txt'
df = parse_to_df(data_path=data_path, size_to_read=5 * 1024 ** 2)
print(df.dtypes)
df.to_csv(data_path / 'train.csv')

