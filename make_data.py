import numpy as np
import pandas as pd


n_cols = 10
df = pd.DataFrame(
    columns=['C{}'.format(i) for i in range(n_cols)],
    data=np.random.randint(
        [i for i in range(n_cols)], 
        [5*(i+1) for i in range(n_cols)],
        size=(100, n_cols))
)
print(df.head())
df.to_csv('./data/toy_data.csv', index=False)
