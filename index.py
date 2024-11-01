import pandas as pd

data = pd.read_csv('./filter_data.csv', sep=',')

print("Tipet e të dhënave para rregullimeve:\n", data.dtypes)

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')  # Transformoni në datë
data['Income'] = data['Income'].astype(float)  # Sigurohuni që të jetë float
data['Year_Birth'] = data['Year_Birth'].astype(int)  # Sigurohuni që të jetë int

print("\nTipet e të dhënave pas rregullimeve:\n", data.dtypes)

print("\nDisa rreshta nga dataset-i:\n", data.head())

missing_values = data.isnull().sum()
print("\nVlerat e zbrazëta në çdo kolonë:\n", missing_values)

duplicate_rows = data.duplicated().sum()
print("\nNumri i dyfishimeve:", duplicate_rows)

print("\nStatistikat për shpërndarjen e të dhënave:\n", data.describe())

data['Income'] = data['Income'].fillna(data['Income'].mean())  # Zëvendëso vlerat e zbrazëta me mesataren

data = data.drop_duplicates()

data.to_csv('./cleaned_filter_data.csv', index=False)

print("\nTë dhënat e pastruara janë ruajtur në './cleaned_filter_data.csv'.")
