import pandas as pd
import numpy as np

df = pd.read_csv('Predict_FAT.csv')
# df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

# print(train)
# print(test)

train.to_csv('Predict_FAT_TRAIN.csv', index=False)
test.to_csv('Predict_FAT_TEST.csv', index=False)
