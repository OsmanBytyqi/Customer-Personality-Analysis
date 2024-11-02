import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./processed_data.csv', sep=',')


print(data.dtypes)
print(data.describe())

##
### Agregimi
##data['Total_Mnt'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                          ##'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
##
##
### Mostrimi
##data = data[(data['Income'] >= 0) & (data['MntWines'] >= 0) & (data['MntFruits'] >= 0)]
##
##
##
### Kontrollo për mungesa
##missing_data = data.isnull().sum()
##print(missing_data[missing_data > 0])  # Shfaq vetëm kolonat me vlera të zbrazëta
##
### Plotëso mungesat në kolonën 'Income' me mesataren e saj si shembull
##data['Income'].fillna(data['Income'].mean(), inplace=True)
##
##
##
### Shembull për zgjedhje të vetive të caktuara
##selected_features = data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome',
                          ##'Recency', 'NumDealsPurchases', 'Total_Mnt']]
##
##
##data['Family_Size'] = data['Kidhome'] + data['Teenhome']
##
##
##data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                              ##labels=['Low', 'Medium', 'High', 'Very High'])
##
### Normalizo kolonën 'Income'
##scaler = StandardScaler()
##data['Income'] = scaler.fit_transform(data[['Income']])
##
##
##
##data = pd.get_dummies(data, columns=['Marital_Status'])
