import pandas as pd
import numpy as np

df = pd.read_feather("./scores.feather")
print(df.info())

df['norm_score'] = np.log(df['score'])
mini = df['norm_score'].min()
maxi = df['norm_score'].max()

df['norm_score'] = (df['norm_score'] - mini) / (maxi - mini)


print(df.head())
df.to_feather("norm_scores.feather")
