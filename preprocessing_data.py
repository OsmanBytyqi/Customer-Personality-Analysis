import pandas as pd

# Read the data from the input file
data = pd.read_csv('./raw_data.csv', sep=',')

# Print out the information of the columns and a summary of the statistics to understand the structure.
#print(data.dtypes)



duplicates_specific = data[data.duplicated(subset=['Year_Birth'])]
print("Duplicated rows based on specific columns (Year_Birth):")
print(duplicates_specific)

#print(data.describe())

# Statistical summary for the 'Income' column
#print(data['Income'].describe())

# Convert the data to more appropriate formats
data['Education'] = data['Education'].astype('category')
data['Marital_Status'] = data['Marital_Status'].astype('category')
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

# Check for null values
#print(data.isnull().sum())

# Handle missing values in the 'Income' column
# Since we have confirmed there are 24 missing values in 'Income':
median_income = data['Income'].median()
data['Income'] = data['Income'].fillna(median_income)


# Create a new feature for total spending
data['Total_Mnt'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Data we want for further analyze  Columns: 'Year_Birth', 'Income', 'Kidhome',
# 'Teenhome', 'Recency', 'NumDealsPurchases', 'Total_Mnt'
selected_features = data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome',
                          'Recency', 'NumDealsPurchases', 'Total_Mnt']]

# Discretization of the 'Income' column into 4 groups
data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                              labels=['Low', 'Medium', 'High', 'Very High'])

data.to_csv('cleaned_data.csv', index=False)

#print(data.dtypes)

