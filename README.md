# Customer Data Analysis Project

This project involves analyzing a dataset of customer data to gain insights and identify patterns. The project covers data preprocessing, exploratory analysis, outlier detection, and dimensionality reduction using Principal Component Analysis (PCA).

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Outlier Detection](#outlier-detection)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset (`raw_data.csv`) contains demographic and purchasing information about customers, including their age, income, and spending on various product categories.

**Features include:**
- `Year_Birth`: Year the customer was born.
- `Education`: Level of education.
- `Marital_Status`: Marital status of the customer.
- `Income`: Annual income of the customer.
- `Dt_Customer`: Date when the customer joined.
- `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`: Spending amounts on various products.
- `Kidhome`, `Teenhome`: Number of children/teens in the customer’s home.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/customer-data-analysis.git
    ```
2. Navigate to the project directory:
    ```bash
    cd customer-data-analysis
    ```

3. Add an Environment Setup Step (optional)
    ```bash
    python -m venv env
    source env/bin/activate
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

1. **Duplicate Removal**: Duplicates were removed based on the `Year_Birth` column.
2. **Data Type Conversion**:
   - Converted `Education` and `Marital_Status` to categorical types.
   - Converted `Dt_Customer` to datetime format.
3. **Handling Missing Values**:
   - Filled missing values in the `Income` column with the median income.
4. **Feature Engineering**:
   - Created a `Total_Mnt` feature to represent the total spending on all products.
   - Calculated `Age` from `Year_Birth`.
   - Created a `Family_Size` feature based on `Kidhome` and `Teenhome`.

## Exploratory Data Analysis

Several analyses were performed to understand data distribution and feature relationships:

1. **Correlation Matrix**: Analyzed correlations between selected features to identify potential relationships.
2. **Income Grouping**: Discretized `Income` into categories (`Low`, `Medium`, `High`, `Very High`) for further analysis.
3. **Distribution Analysis**:
   - **Age Distribution**: Plotted the distribution of customer age.
   - **Income Group Distribution**: Analyzed income distribution among different customer groups.
   - **Family Size vs. Total Spending**: Analyzed spending behavior across family sizes.
   - **Income vs. Total Spending**: Visualized spending patterns relative to income.

## Outlier Detection

Outliers in the `Income` column were detected using the Interquartile Range (IQR) method:
- Defined outliers as values outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
- Displayed the outliers to assess their impact on the analysis.

## Principal Component Analysis (PCA)

To reduce the dimensionality and visualize the data:

1. **Standardization**: Scaled the selected features to have a mean of 0 and a standard deviation of 1.
2. **PCA Transformation**:
   - Reduced the data to 2 principal components for visualization.
   - Calculated the explained variance ratio to understand the proportion of variance captured by each component.
3. **Visualization**:
   - Created a scatter plot to visualize the PCA result, color-coded by `Income_Group`.

## Results

- **Correlation Analysis**: Key features such as `Income` and `Total_Mnt` were found to be correlated.
- **PCA Analysis**: The first two principal components capture a significant portion of the variance, simplifying visualization without losing much information.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
