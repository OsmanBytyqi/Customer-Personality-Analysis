# Customer Data Analysis Project

This project involves analyzing a dataset of customer data to gain insights and
identify patterns. The project covers data preprocessing, exploratory analysis,
outlier detection, and dimensionality reduction using Principal Component
Analysis (PCA).

## University and Contributors
- **University**: University of Pristina
- **Faculty**: Faculty of Electrical and Computer Engineering
- **Mentor**: Dr. Sc. Mërgim H. HOTI
- **Students**:
  - Fisnik MUSTAFA
  - Osman BYTYQI
  - Urim HOXHA

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

3. Add an Environment Setup (optional)
   ```bash
   python -m venv env
   ```
4. Activate the Environment
   - **macOS/Linux**: Use `source env/bin/activate` to activate the virtual environment.
   - **Windows**: Use `env\Scripts\activate` to activate the virtual environment.
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

1. **Duplicate Removal**: Checked for duplicates to remove.
2. **Data Type Conversion**:
   - Converted `Dt_Customer` to datetime format.
3. **Handling Missing Values**:
   - Filled missing values in the `Income` column with the median income.
4. **Feature Engineering**:
   - Created a `Total_Mnt` feature to represent the total spending on all products.
   - Calculated `Age` from `Year_Birth`.
   - Created a `Family_Size` feature based on `Kidhome` and `Teenhome`.
   - Created a `Customer_Tenure` feature based on `Dt_customer`
5. **Income Grouping**: Discretized `Income` into categories (`Low`, `Medium`, `High`, `Very High`) for further analysis.

## Project Structure

The following structure explains the files and folders included in this project:

```
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── LICENSE                     # Contains the license information for the project
├── README.md                   # Documentation and overview of the project
├── preprocessed_data.csv       # Dataset after preprocessing
├── preprocessing_data.py       # Script for handling data preprocessing
├── raw_data.csv                # Raw, unprocessed dataset
├── requirements.txt            # List of Python dependencies for the project
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
