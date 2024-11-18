from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.2f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, test_pred))

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred)

def perform_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
                               cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv("data/train.csv")
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    data.fillna(data.median(), inplace=True)

    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform grid search
    best_model = perform_grid_search(X_train, y_train)

    # Evaluate the model
    train_acc, test_acc = evaluate_model(best_model, X_train, y_train, X_test, y_test)

    # Save the best model
    joblib.dump(best_model, "best_model.pkl")
