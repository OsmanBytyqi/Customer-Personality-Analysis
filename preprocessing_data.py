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

# Read the data
data = pd.read_csv('./raw_data.csv', sep=',')

# Remove duplicates based on 'Year_Birth'
duplicates_specific = data[data.duplicated(subset=['Year_Birth'])]
print("Duplicated rows based on specific columns (Year_Birth):")
print(duplicates_specific)

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

# Normalizing the 'Total_Mnt' column to a new range (0 to 100) with two decimal places

# Min and Max values of 'Total_Mnt'
min_total_mnt = data['Total_Mnt'].min()
max_total_mnt = data['Total_Mnt'].max()

# Apply normalization
data['Total_Mnt_Normalized'] = ((data['Total_Mnt'] - min_total_mnt) / (max_total_mnt - min_total_mnt)) * 100
data['Total_Mnt_Normalized'] = data['Total_Mnt_Normalized'].round(2)

# Display the first few rows of 'Total_Mnt' and 'Total_Mnt_Normalized' for verification
print(data[['Total_Mnt', 'Total_Mnt_Normalized']].head())

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
data.to_csv('preprocessed_data.csv', index=False)

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
sns.histplot(data['Total_Mnt_Normalized'], kde=True, color='green')
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
numeric_cols = data.select_dtypes(include=[np.number]).columns.drop(['ID', 'Z_CostContact', 'Z_Revenue'])
numeric_data = data[numeric_cols]
scaled_data = scaler.fit_transform(numeric_data)
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
