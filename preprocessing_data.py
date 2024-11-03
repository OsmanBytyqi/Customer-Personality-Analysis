import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.gridspec as gridspec
import seaborn as sns

pd.set_option('display.max_columns', None)

# Read the data
data = pd.read_csv('./raw_data.csv', sep=',')

# Check some objects of the dataset
print(data.head())

# Check the shape of the dataset
print(data.shape)

# Get general info about the attributes
print(data.info)

# Get some basic statistics
print(data.describe())

## Data Quality
#----------------
# Check data types
print(data.dtypes)

# Convert data types
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

# Check for missing values
missing_values = data.isnull().sum()
missing_percentage = (missing_values / data.shape[0]) * 100

missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})

# filter to show only columns with missing values
print(missing_info[missing_info['Missing Values'] > 0])

# Handle missing values
median_income = data['Income'].median()
data['Income'] = data['Income'].fillna(median_income)

# Check for duplicates
if data.duplicated().any():
    data.drop_duplicates(inplace=True)
else:
    print("No duplicates found.")

# Check if there are any negative values in numerical attributes
numerical_columns = [
    'Income', 'Kidhome', 'Teenhome', 'Recency',	'MntWines',	'MntFruits', 'MntMeatProducts',	'MntFishProducts',	'MntSweetProducts',	'MntGoldProds',	'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

# Count negative values in each of these columns
negative_values_check = {col: (data[col] < 0).sum() for col in numerical_columns}
print(negative_values_check)

# Check the frequency of values in 'Marital_Status'
marital_status_counts = data['Marital_Status'].value_counts()
print(marital_status_counts)

# Replace Marital Status: Alone -> Single; (YOLO, Absurd) -> Other
data['Marital_Status'] = data['Marital_Status'].replace({
    'Alone': 'Single',
    'YOLO': 'Other',
    'Absurd': 'Other',
})
print(data['Marital_Status'].value_counts())

# Replace Education level: 2n Cycle -> Master
data['Education'] = data['Education'].replace('2n Cycle', 'Master')


## Aggregation
#-----------------------
# Create a new feature for total spending
data['Total_Mnt'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
data['Accepted'] = (data[campaign_columns].sum(axis=1) > 0).astype(int)

## Dimensionality Reduction
# remove ID as it does not provide any meaningful information
# remove Z_CostContact and Z_Revenue as they have the same value for all records
data.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], inplace=True)
data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], inplace=True)
data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'], inplace=True)
print(data.columns)


## Feature Engineering
# Add age calculation
current_year = datetime.now().year
data['Age'] = current_year - data['Year_Birth']

# Create Family Size feature
data['Family_Size'] = data['Kidhome'] + data['Teenhome']
data.drop(columns=['Kidhome', 'Teenhome'], inplace=True)

# Calculate 'Customer tenure'
data['Customer_Tenure'] = (datetime.now() - data['Dt_Customer']).dt.days

data.drop(columns=['Year_Birth'], inplace=True)
data.drop(columns=['Dt_Customer'], inplace=True)

print(data[['Age', 'Family_Size', 'Customer_Tenure']].head())


## Discretization
#-------------------
age_bins = [0, 35, 55, 100]
age_labels = ['Young', 'Middle-aged', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Discretize the Income column into groups
data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                               labels=['Low', 'Medium', 'High', 'Very High'])

print(data[['Age_Group', 'Income_Group']].head())

# Normalizing the 'Total_Mnt' column to a new range (0 to 100) with two decimal places

# Min and Max values of 'Total_Mnt'
min_total_mnt = data['Total_Mnt'].min()
max_total_mnt = data['Total_Mnt'].max()

# Apply normalization
data['Total_Mnt_Normalized'] = ((data['Total_Mnt'] - min_total_mnt) / (max_total_mnt - min_total_mnt)) * 100
mnt_normalized = data['Total_Mnt_Normalized'] = data['Total_Mnt_Normalized'].round(2)
#data.drop(columns=['Total_Mnt_Normalized'], inplace=True)


# Display the first few rows of 'Total_Mnt' and 'Total_Mnt_Normalized' for verification
print(data[['Total_Mnt', 'Total_Mnt_Normalized']].head())

# Save cleaned data
data.to_csv('preprocessed_data.csv', index=False)