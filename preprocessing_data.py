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
data['Accepted'] = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].any(axis=1).astype(int)


## Dimensionality Reduction
# remove ID as it does not provide any meaningful information
# remove Z_CostContact and Z_Revenue as they have the same value for all records
data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], inplace=True)# Drop the original campaign columns
data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], inplace=True)
print(data.columns)


## Feature Engineering
# Add age calculation
current_year = datetime.now().year
data['Age'] = current_year - data['Year_Birth']

# Create Family Size feature
data['Family_Size'] = data['Kidhome'] + data['Teenhome']

# Calculate 'Customer tenure'
data['Customer_Tenure'] = (datetime.now() - data['Dt_Customer']).dt.days

print(data[['Year_Birth', 'Age', 'Kidhome', 'Teenhome', 'Family_Size', 'Customer_Tenure']].head())
data.drop(columns=['Year_Birth'])

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
data['Total_Mnt_Normalized'] = data['Total_Mnt_Normalized'].round(2)
total_Mnt_Normalized = data['Total_Mnt_Normalized']

# Display the first few rows of 'Total_Mnt' and 'Total_Mnt_Normalized' for verification
print(data[['Total_Mnt', 'Total_Mnt_Normalized']].head())
data.drop(columns=['Total_Mnt_Normalized'])


# Selected features for analysis
selected_features = data[['Income', 'Kidhome', 'Teenhome',
                          'Recency', 'NumDealsPurchases', 'Total_Mnt', 'Age', 'Family_Size']]


# Save cleaned data
data.to_csv('preprocessed_data1.csv', index=False)

# Analysis of Selected Features
print(selected_features.describe())

# IQR Method for Income
Q1 = data['Income'].quantile(0.25)
Q3 = data['Income'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
outliers = data[(data['Income'] < lower_bound) | (data['Income'] > upper_bound)]
print("Outliers in Income:")
print(outliers[['Income']])


# Correlation Analysis
correlation_matrix = selected_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Income Group Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Income_Group', palette='Set2')
plt.title('Distribution of Income Groups')
plt.xlabel('Income Group')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=list(range(0, 100, 5)), kde=True, color='blue')  # Define bins as a list
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Family Size vs. Total Spending
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Family_Size', y='Total_Mnt', palette='Set1')
plt.title('Total Spending by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Total Spending')
plt.show()

# Income vs. Total Spending
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='Total_Mnt', hue='Income_Group', palette='Set2', alpha=0.7)
plt.title('Income vs. Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.legend(title='Income Group')
plt.show()

# Recency and Spending
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Recency', y='Total_Mnt', marker='o', ci=None)
plt.title('Total Spending by Recency')
plt.xlabel('Recency (days since last purchase)')
plt.ylabel('Total Spending')
plt.grid()
plt.show()

# Plot the distribution of the min-max normalized Total_Mnt column (0 to 100 range)
plt.figure(figsize=(10, 6))
sns.histplot(total_Mnt_Normalized, kde=True, color='green')
plt.title('Distribution of Min-Max Normalized Total Spending (0 to 100 range)')
plt.xlabel('Total_Mnt_Normalized (0-100)')
plt.ylabel('Frequency')
plt.show()

# PCA Implementation
# Standardizing the selected features
scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

# Apply PCA
pca = PCA(n_components=2)  # Change this to retain more components if needed
principal_components = pca.fit_transform(selected_features_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, data[['Income_Group']].reset_index(drop=True)], axis=1)  # Add income group
# Visualize PCA Results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Income_Group', palette='Set2', alpha=0.7)
plt.title('PCA Result: PC1 vs PC2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Income Group')
plt.grid()
plt.show()

sample_data = data.sample(n=500, random_state=42)

# For example, plot the distribution of 'Income' in this sample
plt.figure(figsize=(10, 6))
sns.histplot(sample_data['Income'], kde=True, color='blue')
plt.title('Income Distribution in Sample of 500 Rows')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Selecting numerical columns for clustering analysis, excluding ID and constant columns
# numeric_cols = data.select_dtypes(include=[np.number]).columns.drop(['ID', 'Z_CostContact', 'Z_Revenue'])
# numeric_data = data[numeric_cols]
scaled_data = scaler.fit_transform(data)
reduced_data = pca.fit_transform(scaled_data)

# Apply DBSCAN for clustering and noise detection
dbscan = DBSCAN(eps=0.5, min_samples=10)  # eps and min_samples can be fine-tuned
clusters = dbscan.fit_predict(reduced_data)

# Adding the cluster labels to the data
data['Cluster'] = clusters

# Counting and displaying the number of outliers (labelled as -1 by DBSCAN)
outliers_count = (clusters == -1).sum()
outliers_count, data['Cluster'].value_counts()

outliers_data = data[data['Cluster'] == -1]
non_outliers_data = data[data['Cluster'] == 0]

# Plotting the clusters with the identified outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette="viridis", legend="full", s=50)
plt.title("DBSCAN Clustering with Outliers")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

# Set up the figure and subplots
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig)

# Plot Income distribution
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(non_outliers_data['Income'], color='blue', label='Non-Outliers', kde=True, ax=ax1)
sns.histplot(outliers_data['Income'], color='red', label='Outliers', kde=True, ax=ax1)
ax1.set_title('Income Distribution')
ax1.legend()

# Plot Total Spending distribution
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(non_outliers_data['Total_Mnt'], color='blue', label='Non-Outliers', kde=True, ax=ax2)
sns.histplot(outliers_data['Total_Mnt'], color='red', label='Outliers', kde=True, ax=ax2)
ax2.set_title('Total Spending Distribution')
ax2.legend()

# Plot Recency (days since last purchase) distribution
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(non_outliers_data['Recency'], color='blue', label='Non-Outliers', kde=True, ax=ax3)
sns.histplot(outliers_data['Recency'], color='red', label='Outliers', kde=True, ax=ax3)
ax3.set_title('Recency Distribution')
ax3.legend()

# Plot campaign acceptance rates (mean across campaigns)
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
outliers_data['Mean_Accepted'] = outliers_data[campaign_cols].mean(axis=1)
non_outliers_data['Mean_Accepted'] = non_outliers_data[campaign_cols].mean(axis=1)

ax4 = fig.add_subplot(gs[1, 1])
sns.histplot(non_outliers_data['Mean_Accepted'], color='blue', label='Non-Outliers', kde=True, ax=ax4)
sns.histplot(outliers_data['Mean_Accepted'], color='red', label='Outliers', kde=True, ax=ax4)
ax4.set_title('Campaign Acceptance Rate')
ax4.legend()

plt.show()

# Displaying the first few rows of outlier data for inspection
outliers_data = data[data['Cluster'] == -1]
outliers_data.head()

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each principal component: {explained_variance}")
