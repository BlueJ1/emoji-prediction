from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def pca(df):
    df.columns = df.columns.astype(str)

    print('Performing PCA...')
    pca_object = PCA(n_components=3)
    reduced_dim_df = pca_object.fit_transform(df)
    plot = plot1(reduced_dim_df, df)

    # save plot to png file
    plot.savefig('pca.png')

    return reduced_dim_df


def plot1(reduced_dim_df, df):
    # plot such that each emoji is a different color
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_dim_df[:, 0], reduced_dim_df[:, 1],
               reduced_dim_df[:, 2], c=df['emoji'])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    return plt


def plot2(reduced_dim_df, df):

    # Creating a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(reduced_dim_df[:, 0], reduced_dim_df[:, 1],
               reduced_dim_df[:, 2], s=30, alpha=0.6, edgecolors='w')

    # color each datapoint according to emoji label
    sns.scatterplot(
        x=reduced_dim_df[:, 0], y=reduced_dim_df[:, 1], hue=df['emoji'],
        palette=sns.color_palette("hls", 5),
        legend="full", alpha=0.3, edgecolor='w')

    # Adding labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Plot')

    # Show the plot
    plt.show()
