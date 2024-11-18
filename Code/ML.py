from data_processing import load_data, clean_data
from data_visualization import plot_perished_distribution, plot_pclass_distribution, plot_heatmap
from model_training import train_model, evaluate_model, perform_grid_search, evaluate_best_model
import os

if __name__ == "__main__":
    # Paths
    path = "data"
    graph_path = 'Graphs'

    # Load and Clean Data
    df, df_test = load_data(path)
    df = clean_data(df)

    # Visualize Data
    plot_perished_distribution(df, graph_path)
    plot_pclass_distribution(df, graph_path)
    plot_heatmap(df, graph_path)

    # Train and Evaluate Model
    X = df.drop('Perished', axis=1)
    y = df['Perished']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initial Model Training and Evaluation
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Perform Grid Search and Evaluate Best Model
    grid_search = perform_grid_search(X_train, y_train)
    evaluate_best_model(grid_search, X_test, y_test)
