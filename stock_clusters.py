import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_stock_clusters(df, feature_cols, k=3, use_std=False):
    df_cluster = df.groupby('Symbol')[feature_cols].mean().reset_index()

    if use_std and 'ClosePrice' in feature_cols:
        close_std = df.groupby('Symbol')['ClosePrice'].std().rename('CloseStd')
        df_cluster = df_cluster.merge(close_std, on='Symbol', how='left')
        feature_cols.append('CloseStd')

    df_cluster = df_cluster.dropna()

    X_scaled = StandardScaler().fit_transform(df_cluster[feature_cols])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df_cluster['PC1'], df_cluster['PC2'] = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_cluster, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
    ax.set_title('Stock Clusters (PCA Projection)')
    ax.grid(True)
    return fig, df_cluster
