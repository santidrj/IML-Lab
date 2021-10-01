from scipy.io import arff
import pandas as pd

data = arff.loadarff('../datasets/datasets/adult.arff')
df = pd.DataFrame(data[0])
print(data[1])
print(df.head())
print(df.dtypes)
