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

# Set display options
pd.set_option('display.max_columns', None)

# Read the data
data = pd.read_csv('./raw_data.csv', sep=',')

# Check some objects of the dataset
print(data.head())

# Check the shape of the dataset
print(data.shape)

# Get general info about the attributes
print(data.info())

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

# Filter to show only columns with missing values
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
    'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

# Count negative values in each of these columns
negative_values_check = {col: (data[col] < 0).sum() for col in numerical_columns}
print(negative_values_check)

# Check the frequency of values in 'Marital_Status'
marital_status_counts = data['Marital_Status'].value_counts()
print(marital_status_counts)

# Reduce the Relationship Status to two options: Single, Partnered
data['Relationship_Status'] = data['Marital_Status'].replace({
    'Alone': 'Single', 'YOLO': 'Single', 'Absurd': 'Single', 'Divorced': 'Single', 'Widow': 'Single',
    'Married': 'Partnered', 'Together': 'Partnered'
})
print(data['Relationship_Status'].value_counts())

# Replace Education level: 2n Cycle -> Master
data['Education'] = data['Education'].replace({
    'Basic': 'High School', 'Graduation': 'Bachelor', '2n Cycle': 'Master'
})
print(data['Education'].value_counts())


## Aggregation
#-----------------------
# Create a new feature for total spending
data['Total_Spent'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                             'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
data['Accepted'] = (data[campaign_columns].sum(axis=1) > 0).astype(int)

## Dimensionality Reduction
# Remove ID as it does not provide any meaningful information
# Remove Z_CostContact and Z_Revenue as they have the same value for all records
data.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                   'MntSweetProducts', 'MntGoldProds'], inplace=True)
data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Marital_Status'], inplace=True)
data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'], inplace=True)
print(data.columns)


## Feature Engineering
# Add age calculation
current_year = datetime.now().year
data['Age'] = current_year - data['Year_Birth']

# Create Family Size feature
data['Family_Size'] = data['Relationship_Status'].replace({'Single': 1, 'Partnered': 2}) + data['Kidhome'] + data['Teenhome']
data.drop(columns=['Kidhome', 'Teenhome'], inplace=True)

# Calculate 'Customer tenure'
data['Customer_Tenure'] = (datetime.now() - data['Dt_Customer']).dt.days

data.drop(columns=['Year_Birth'], inplace=True)
data.drop(columns=['Dt_Customer'], inplace=True)

print(data[['Age', 'Family_Size', 'Customer_Tenure']].head())

# Encoding Categorical Attributes
# Label encoding for Education
education_order = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3,
}
data['Education'] = data['Education'].replace(education_order)

# One-hot encoding for Relationship Status
data = pd.get_dummies(data, columns=['Relationship_Status'])

# Function to handle outliers using IQR and capping for all numeric fields
def cap_outliers_with_iqr(df, columns):
    for col in columns:
        # Calculate Q1 and Q3 for each numeric column
        #
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers while maintaining original type
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype(int)
        else:  # Float or other numeric types
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df



# Identify all numeric columns in the dataset
numeric_columns = data.select_dtypes(include=[np.number]).columns


for selected_column in numeric_columns:
    # Create a subplot grid (1 row, 2 columns for before and after)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot before handling outliers (original data)
    sns.boxplot(data[selected_column], ax=axes[0])
    axes[0].set_title(f'{selected_column} (Before Handling Outliers)')

    # Apply IQR capping directly to the 'data' dataframe for the selected column
    cap_outliers_with_iqr(data, [selected_column])  # Directly modify the data dataframe
    sns.boxplot(data[selected_column], ax=axes[1])
    axes[1].set_title(f'{selected_column} (After Handling Outliers with Capping)')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()
    plt.show()

## Discretization
#-------------------
age_bins = [0, 35, 55, 100]
age_labels = ['Young', 'Middle-aged', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Discretize the Income column into groups
data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                               labels=['Low', 'Medium', 'High', 'Very High'])

print(data[['Age_Group', 'Income_Group']].head())

# Normalize the 'Total_Spent' column to a new range (0 to 100) with two decimal places
# Min and Max values of 'Total_Spent'
min_Total_Spent = data['Total_Spent'].min()
max_Total_Spent = data['Total_Spent'].max()

# Apply normalization
data['Total_Spent_Normalized'] = ((data['Total_Spent'] - min_Total_Spent) / (max_Total_Spent - min_Total_Spent)) * 100
data['Total_Spent_Normalized'] = data['Total_Spent_Normalized'].round(2)

# Display the first few rows of 'Total_Spent' and 'Total_Spent_Normalized' for verification
print(data[['Total_Spent', 'Total_Spent_Normalized']].head())

# Drop 'Total_Spent' as it was only used for the calculation
data.drop(columns=['Total_Spent_Normalized'], inplace=True)

# Save cleaned data
data.to_csv('preprocessed_data.csv', index=False)
