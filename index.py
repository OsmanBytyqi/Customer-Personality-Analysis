import pandas as pd

data = pd.read_csv('./raw_data.csv', sep=',')




#print(data.columns)

#print(data.head())
##print(data.info())
print(data.dtypes)
#print(data.describe())

# sampled_data = data.sample(frac=0.1)
