# Customer Data Analysis Project

This project involves analyzing a dataset of customer data to gain insights and
identify patterns. The project covers data preprocessing, exploratory analysis,
outlier detection, and dimensionality reduction using Principal Component
Analysis (PCA).

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Contributing](#contributing)

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
   - Converted `Dt_Customer` to datetime format.
3. **Handling Missing Values**:
   - Filled missing values in the `Income` column with the median income.
4. **Feature Engineering**:
   - Created a `Total_Mnt` feature to represent the total spending on all products.
   - Calculated `Age` from `Year_Birth`.
   - Created a `Family_Size` feature based on `Kidhome` and `Teenhome`.
   - Created a `Customer_Tenure` feature based on `Dt_customer`

## Exploratory Data Analysis
Several analyses were performed to understand data distribution and feature relationships:

1. **Income Grouping**: Discretized `Income` into categories (`Low`, `Medium`, `High`, `Very High`) for further analysis.

## Outlier Detection

- Displayed the outliers to assess their impact on the analysis.

## Results

- **Correlation Analysis**: Key features such as `Income` and `Total_Mnt` were found to be correlated.
- **PCA Analysis**: The first two principal components capture a significant portion of the variance, simplifying visualization without losing much information.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
