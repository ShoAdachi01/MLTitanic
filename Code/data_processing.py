import pandas as pd
import os

def load_data(path):
    # Load train and test datasets
    df = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))
    return df, df_test

def clean_data(df, is_train=True, feature_names=None):
    # Fill missing 'Age' with median
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Drop the 'Cabin' column due to high number of missing values
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    # Fill missing 'Embarked' values with the mode
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Drop unnecessary columns
    if "Name" in df.columns and "Ticket" in df.columns:
        df.drop(columns=["Name", "Ticket"], inplace=True)

    # For test data, align columns to training feature names
    if not is_train:
        df = df.reindex(columns=feature_names, fill_value=0)

    return df

if __name__ == "__main__":
    path = "data"

    # Load train and test data
    df_train, df_test = load_data(path)

    # Process train data
    df_train = clean_data(df_train)

    # Save feature names from the training dataset
    feature_names = list(df_train.drop(columns=["Perished"]).columns)
    pd.Series(feature_names).to_csv("feature_names.csv", index=False)

    # Process test data, aligning it with training feature names
    df_test = clean_data(df_test, is_train=False, feature_names=feature_names)

    # Debug: Print information
    print("Training data after cleaning:")
    print(df_train.info())
    print("\nTest data after cleaning:")
    print(df_test.info())
