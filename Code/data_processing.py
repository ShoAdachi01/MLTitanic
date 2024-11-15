import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(path, 'test.csv'))
    return df, df_test

def clean_data(df):
    # Fill missing 'Age' with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Drop the 'Cabin' column due to high number of missing values
    df.drop(columns=['Cabin'], inplace=True)
    # Fill missing 'Embarked' values with the mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    # Drop unnecessary columns
    df.drop(columns=['Name', 'Ticket'], inplace=True)
    return df

if __name__ == "__main__":
    path = "data"
    df, df_test = load_data(path)
    df = clean_data(df)
    print(df.info())
