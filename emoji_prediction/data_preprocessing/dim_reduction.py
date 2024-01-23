from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def pca(df):
    """
    This function applies Principal Component Analysis (PCA) to reduce the dimensionality of the given DataFrame.
    It reduces the dimensionality to 3 and plots the result in a 3D scatter plot.
    It also calculates and prints the silhouette score of the reduced DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to apply PCA to.

    Returns:
    ndarray: The reduced DataFrame as a numpy array.
    """
    df.columns = df.columns.astype(str)
    print('Performing PCA...')
    pca_object = PCA(n_components=3)
    reduced_dim_df = pca_object.fit_transform(df)
    plot(reduced_dim_df, df, name='PCA_Plot')
    score_pca = silhouette_score(reduced_dim_df, df['emoji'])
    print('Silhouette score: ' + str(score_pca))

    return reduced_dim_df


def plot(reduced_dim_df, df, name="Plot"):
    """
    This function plots the given reduced DataFrame in a 3D scatter plot.
    Each point in the plot is colored according to its corresponding emoji in the original DataFrame.

    Parameters:
    reduced_dim_df (ndarray): The reduced DataFrame as a numpy array.
    df (DataFrame): The original DataFrame.
    name (str, optional): The title of the plot. Defaults to "Plot".
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_dim_df[:, 0], reduced_dim_df[:, 1],
               reduced_dim_df[:, 2], c=df['emoji'])
    ax.set_title(name)
    plt.savefig(name + '.png')
    plt.show()


def t_sne(df):
    """
    This function applies t-SNE to reduce the dimensionality of the given DataFrame.
    It reduces the dimensionality to 3 and plots the result in a 3D scatter plot.
    It also calculates and prints the silhouette score of the reduced DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to apply t-SNE to.

    Returns:
    ndarray: The reduced DataFrame as a numpy array.
    """
    print('Performing t-SNE...')
    tsne = TSNE(n_components=3, verbose=1, perplexity=500, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    plot(tsne_results, df, name='t-SNE_Plot')
    score_tsne = silhouette_score(tsne_results, df['emoji'])
    print('Silhouette score: ' + str(score_tsne))
    return tsne_results
