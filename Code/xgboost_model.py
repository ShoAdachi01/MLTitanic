from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_xgboost(X_train, y_train):
    # Define the XGBoost model with basic parameters
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Evaluate the model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.2f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, test_pred))

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred)

def perform_grid_search_xgboost(X_train, y_train):
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # GridSearchCV for XGBoost
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Load and preprocess the data
    data = pd.read_csv("data/train.csv")
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    data.fillna(data.median(), inplace=True)

    X = data.drop("Perished", axis=1)
    y = data["Perished"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform grid search for XGBoost
    best_xgb_model = perform_grid_search_xgboost(X_train, y_train)

    # Evaluate the model
    train_acc, test_acc = evaluate_model(best_xgb_model, X_train, y_train, X_test, y_test)

    # Save the best model
    joblib.dump(best_xgb_model, "best_xgboost_model.pkl")
    print("Model saved as 'best_xgboost_model.pkl'")
