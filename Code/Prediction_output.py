import pandas as pd
import joblib
import numpy as np

if __name__ == "__main__":
    # Paths
    path = "data"

    # Load submission format file
    submission = pd.read_csv(path + "/gender_submission.csv")

    # Load the test dataset
    test_data = pd.read_csv(path + "/test.csv")

    # Keep PassengerId for submission file
    passenger_ids = test_data["PassengerId"]

    # Preprocess test data
    test_data = test_data.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors="ignore")  # Drop irrelevant columns
    test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked"], drop_first=True)  # One-hot encode
    test_data = test_data.fillna(test_data.median())  # Fill missing values

    # Load feature names from training
    feature_names = pd.read_csv("feature_names.csv", header=None).squeeze().tolist()

    # Debugging: Check features in feature_names and test_data
    print("Features in feature_names.csv:")
    print(feature_names)

    print("\nColumns in test_data before alignment:")
    print(test_data.columns.tolist())

    # Align test data columns with training data
    test_data = test_data.reindex(columns=feature_names, fill_value=0)  # Add missing columns and ensure order

    # Debugging: Check columns after alignment
    print("\nColumns in test_data after alignment:")
    print(test_data.columns.tolist())

    # Debugging: Check if the number of columns matches the model's expected input
    print(f"\nTest data shape: {test_data.shape}")
    print(f"Model expects {joblib.load('best_model.pkl').n_features_in_} features.")

    # Convert test data to NumPy array
    test_data_array = test_data.values

    # Load the trained Random Forest model
    model = joblib.load("best_model.pkl")

    # Make predictions
    predictions = model.predict(test_data_array)

    # Add predictions to the submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Perished": predictions
    })

    # Save the submission file
    submission.to_csv(path + "/submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")
