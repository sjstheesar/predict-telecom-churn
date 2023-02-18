import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    ConfusionMatrixDisplay,
)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, index_col='customerID')
    
    # Check unique values for each column
    for column in df.columns:
        print(f"{column} unique values: {df[column].unique()}")
    
    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remove "(automatic)" from PaymentMethod
    df['PaymentMethod'] = df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
    
    # Identify and handle missing values
    features_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
    print(f"Features with missing values: {features_na}")
    
    # Drop rows with missing TotalCharges
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    return df

# Plot categorical data against target variable
def plot_categorical_to_target(df, categorical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(categorical_values) / 2)
    
    fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(12, 5 * number_of_rows))
    axs = axs.flatten()
    
    for index, column in enumerate(categorical_values):
        sns.countplot(x=column, data=df, hue=target, palette="Blues", ax=axs[index])
        axs[index].set_title(column)
    
    # Remove any extra subplots
    if len(categorical_values) % 2 != 0:
        fig.delaxes(axs[-1])
    
    plt.tight_layout()
    plt.show()

# Plot numerical data against target variable using histograms
def plot_numerical_to_target(df, numerical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values) / 2)
    
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    
    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        sns.kdeplot(df[df[target] == "Yes"][column], fill=True, label="Churn")
        sns.kdeplot(df[df[target] == "No"][column], fill=True, label="No Churn")
        ax.set_title(column)
        ax.legend(loc='upper right')
    
    plt.savefig("numerical_variables.png", dpi=300)
    plt.show()

# Check for outliers using boxplots
def plot_outliers(df, numerical_values):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values) / 2)
    
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    
    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        sns.boxplot(x=column, data=df, palette="Blues")
        ax.set_title(column)
    
    plt.savefig("Outliers_check.png", dpi=300)
    plt.show()

# Encode categorical variables using label encoding
def label_encode(df, features):
    df[features] = df[features].applymap(lambda x: 1 if x == "Yes" else 0)
    return df

# Encode categorical variables using one-hot encoding
def one_hot_encode(df, features):
    return pd.get_dummies(df, columns=features)

# Normalize numerical features using MinMaxScaler
def min_max_normalize(df, features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])
    return df

# Visualize feature importance for a given classifier
def plot_feature_importance(classifier, X_train, y_train, n_features=10):
    if isinstance(classifier, RandomForestClassifier):
        importances = classifier.feature_importances_
    else:
        raise ValueError("Only Random Forest is supported for feature importance visualization.")
    
    feature_names = X_train.columns
    indices = np.argsort(importances)[-n_features:][::-1]
    
    plt.figure(figsize=(7, 6))
    plt.title("Feature Importance")
    sns.barplot(y=feature_names[indices], x=importances[indices])
    plt.show()

# Evaluate model performance using confusion matrix, ROC curve, and precision-recall curve
def evaluate_model(X_train, y_train, X_test, y_test, y_pred, y_pred_proba, classifier_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.show()
    
    print(f"Accuracy Score (Test): {accuracy_score(y_test, y_pred)}")
    print(f"Accuracy Score (Train): {classifier.score(X_train, y_train)}")
    
    y_pred_prob = y_pred_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"{classifier_name}")
    plt.title(f"ROC Curve - {classifier_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    print(f"AUC Score (ROC): {auc_score}")
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    auc_pr_score = auc(recall, precision)
    
    plt.plot(recall, precision, label=f"{classifier_name}")
    plt.title(f"Precision-Recall Curve - {classifier_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    print(f"f1 Score: {f1_score(y_test, y_pred)} \nAUC Score (PR): {auc_pr_score}")

# Main execution
def main():
    file_path = r"C:\Users\Dongen_Master\Desktop\Telecom Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_preprocess_data(file_path)
    
    customer_services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    plot_categorical_to_target(df, customer_services, "Churn")
    
    customer_account_cat = ["Contract", "PaperlessBilling", "PaymentMethod"]
    plot_categorical_to_target(df, customer_account_cat, "Churn")
    
    customer_account_num = ["tenure", "MonthlyCharges", "TotalCharges"]
    plot_numerical_to_target(df, customer_account_num, "Churn")
    
    numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
    plot_outliers(df, numerical_values)
    
    df = label_encode(df, ["Partner", "Dependents", "PhoneService", "PaperlessBilling"])
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    
    features_ohe = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
    df_ohe = one_hot_encode(df, features_ohe)
    
    features_mms = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_normalized = min_max_normalize(df_ohe.drop(columns=features_mms), features_mms)
    
    X = df_normalized
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_rf_proba = rf.predict_proba(X_test)
    
    evaluate_model(X_train, y_train, X_test, y_test, y_pred_rf, y_pred_rf_proba, "Random Forest")
    plot_feature_importance(rf, X_train, y_train)

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_pred_knn_proba = knn.predict_proba(X_test)
    
    evaluate_model(X_train, y_train, X_test, y_test, y_pred_knn, y_pred_knn_proba, "K-Nearest Neighbors")

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    y_pred_logreg_proba = logreg.predict_proba(X_test)
    
    evaluate_model(X_train, y_train, X_test, y_test, y_pred_logreg, y_pred_logreg_proba, "Logistic Regression")

if __name__ == "__main__":
    main()
