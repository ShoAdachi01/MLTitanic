import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_perished_distribution(df, graph_path):
    f, ax = plt.subplots(1, 2, figsize=(18, 8), facecolor='gray')
    df['Perished'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Perished')
    ax[0].set_ylabel('')
    sns.countplot(x='Perished', data=df, ax=ax[1])
    ax[1].set_title('Perished')
    plt.savefig(os.path.join(graph_path, 'Perished.png'))

def plot_pclass_distribution(df, graph_path):
    plt.clf()
    f, ax = plt.subplots(1, 2, figsize=(18, 8), facecolor='gray')
    df['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
    ax[0].set_title('Number of Passengers By Pclass')
    ax[0].set_ylabel('Count')
    sns.countplot(x='Pclass', hue='Perished', data=df, ax=ax[1])
    ax[1].set_title('Pclass: Perished vs Survived')
    plt.savefig(os.path.join(graph_path, 'PerishedPerClass.png'))

def plot_heatmap(df, graph_path):
    plt.clf()
    df_numeric = df.select_dtypes(include=['number'])
    sns.heatmap(df_numeric.corr(), annot=True, cmap='bwr', linewidths=0.2)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.savefig(os.path.join(graph_path, 'Heatmap.png'))

if __name__ == "__main__":
    import pandas as pd
    path = "data"
    graph_path = 'Graphs'
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    plot_perished_distribution(df, graph_path)
    plot_pclass_distribution(df, graph_path)
    plot_heatmap(df, graph_path)
