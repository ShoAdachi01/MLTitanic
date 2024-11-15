from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return y_pred

def perform_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_best_model(grid_search, X_test, y_test):
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_best))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

if __name__ == "__main__":
    import pandas as pd
    path = "data"
    df = pd.read_csv(path + '/train.csv')
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    X = df.drop('Perished', axis=1)
    y = df['Perished']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train initial model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Perform grid search
    grid_search = perform_grid_search(X_train, y_train)
    evaluate_best_model(grid_search, X_test, y_test)

