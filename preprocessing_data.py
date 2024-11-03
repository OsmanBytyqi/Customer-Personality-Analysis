import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the data
data = pd.read_csv('./raw_data.csv', sep=',')

# Remove duplicates based on 'Year_Birth'
#duplicates_specific = data[data.duplicated(subset=['Year_Birth'])]
#print("Duplicated rows based on specific columns (Year_Birth):")
#print(duplicates_specific)

# Convert data types
data['Education'] = data['Education'].astype('category')
data['Marital_Status'] = data['Marital_Status'].astype('category')
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

# Handle missing values
median_income = data['Income'].median()
data['Income'] = data['Income'].fillna(median_income)

# Create a new feature for total spending
data['Total_Mnt'] = data[['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Add age calculation
current_year = pd.to_datetime('today').year
data['Age'] = current_year - data['Year_Birth']

print(data[['Year_Birth', 'Age']].head())

# Create Family Size feature
data['Family_Size'] = data['Kidhome'] + data['Teenhome']

# Selected features for analysis
selected_features = data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome',
                          'Recency', 'NumDealsPurchases', 'Total_Mnt', 'Age', 'Family_Size']]

# Discretize the Income column into groups
data['Income_Group'] = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000],
                              labels=['Low', 'Medium', 'High', 'Very High'])

# Save cleaned data
data.to_csv('cleaned_data.csv', index=False)

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

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each principal component: {explained_variance}")
