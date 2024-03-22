# Telecom Churn Prediction Project

## Description
This project is designed to predict customer churn in telecom services using machine learning techniques. The script `predict-churn-telecom.py` performs data loading, preprocessing, visualization, and model evaluation.

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Functions and Classes
### load_and_preprocess_data
Loads the dataset from a CSV file and performs preprocessing tasks such as handling missing values and converting data types.

### plot_categorical_to_target
Plots categorical features against the target variable to visualize their distribution relative to churn.

### plot_numerical_to_target
Plots numerical features against the target variable using histograms to identify patterns.

### plot_outliers
Checks for outliers in numerical features using boxplots.

### label_encode
Encodes categorical variables using label encoding.

### one_hot_encode
Encodes categorical variables using one-hot encoding.

### min_max_normalize
Normalizes numerical features using MinMaxScaler.

### plot_feature_importance
Visualizes feature importance for a given classifier, specifically Random Forest in this case.

### evaluate_model
Evaluates model performance using confusion matrix, ROC curve, and precision-recall curve.

## Dependencies
To run the script, you need to have Python 3.x installed on your system. Additionally, ensure that the following packages are installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Usage
1. Install dependencies using pip:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
2. Run the script using:
   ```bash
   python predict-churn-telecom.py
   ```

This will execute the script and perform the churn prediction based on the data in `telco-churn-data.csv`.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License.
