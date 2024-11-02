import pandas as pd
from sklearn.preprocessing import StandardScaler

# Lexo dataset-in
data = pd.read_csv('./raw_data.csv', sep=',')

# Shfaq informacionin e kolonave dhe përmbledhjen e statistikave për të kuptuar strukturën
print(data.dtypes)
print(data.describe())


# Agregimi: Shto kolonën 'Total_Mnt' që përfaqëson totalin e shpenzimeve
data['Total_Mnt'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Mostrimi: Filtrimi i të dhënave për të hequr vlerat negative për 'Income', 'MntWines' dhe 'MntFruits'
data = data[(data['Income'] >= 0) & (data['MntWines'] >= 0) & (data['MntFruits'] >= 0)]

# Kontrollo për mungesa në të gjitha kolonat
missing_data = data.isnull().sum()
print("Kolonat me vlera të zbrazëta:")
print(missing_data[missing_data > 0])  # Shfaq vetëm kolonat me mungesa

# Plotëso mungesat në kolonën 'Income' me mesataren e saj si shembull
data['Income'].fillna(data['Income'].mean(), inplace=True)

# Plotëso mungesat për kolonat numerike me mesataren, dhe ato kategorike me moda (më të shpeshtën)

# Përzgjedhje e vetive: Zgjedh disa kolona specifike për analizë
selected_features = data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome',
                          'Recency', 'NumDealsPurchases', 'Total_Mnt']]

# Krijimi i vetive të reja: Shto kolonën 'Family_Size'
data['Family_Size'] = data['Kidhome'] + data['Teenhome']

# Diskretizimi: Krijo një kolonë për grupet e të ardhurave
data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                              labels=['Low', 'Medium', 'High', 'Very High'])

# Normalizimi: Normalizo kolonat numerike
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Binarizimi: Konverto 'Marital_Status' dhe 'Income_Group' në vlera binare
data = pd.get_dummies(data, columns=['Marital_Status', 'Income_Group'], drop_first=True)

# Shfaq përmbledhjen e të dhënave të përpunuara
print(data.head())
print(data.info())



# Ruaj dataset-in e përpunuar në një file të ri CSV
data.to_csv('./processed_data.csv', index=False)

print("Dataset-i i përpunuar është ruajtur në 'processed_data.csv'.")
