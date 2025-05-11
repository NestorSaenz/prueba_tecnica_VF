from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def segment_members(data, n_clusters=4):
    """Agrupa miembros usando K-means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

def plot_segments(df, x, y, hue):
    """Visualiza segmentos con scatter plot."""
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="viridis")
    plt.title("Segmentación por Engagement y CLTV")
    plt.savefig("segments.png")
    plt.show()