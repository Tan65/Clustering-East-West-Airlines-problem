# Clustering-East-West-Airlines-problem
Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters.  Draw the inferences from the clusters obtained.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import plt for plotting dendrogram
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

# Load the airlines data from CSV file
data = pd.read_excel("EastWestAirlines.xlsx")

# Select the relevant features for clustering
features = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=30)
hc_clusters = hc.fit_predict(X_scaled)

# Plot the dendrogram to determine the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.dendrogram(plt.linkage(X_scaled, method='ward'))  # Corrected plt usage
plt.axhline(y=30, color='r', linestyle='--')
plt.show()

# Perform K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=1, min_samples=10)
dbscan_clusters = dbscan.fit_predict(X_scaled)

# Number of clusters formed
num_clusters_hc = len(np.unique(hc_clusters))
num_clusters_kmeans = len(np.unique(kmeans_clusters))
num_clusters_dbscan = len(np.unique(dbscan_clusters)) - 1  # Subtract 1 to account for noise points

print("Number of clusters formed (Hierarchical Clustering):", num_clusters_hc)
print("Number of clusters formed (K-means Clustering):", num_clusters_kmeans)
print("Number of clusters formed (DBSCAN Clustering):", num_clusters_dbscan)

# Calculate the mean values of each feature for each cluster in K-means clustering
cluster_means = pd.DataFrame(X)
cluster_means['Cluster'] = kmeans_clusters
cluster_means = cluster_means.groupby('Cluster').mean()

# Display the mean values of each feature for each cluster
print("\nMean Values of Features for Each Cluster (K-means Clustering):")
print(cluster_means)

# Calculate the frequency distributions of categorical features for each cluster in K-means clustering
categorical_features = ['cc1_miles', 'cc2_miles', 'cc3_miles']
for feature in categorical_features:
    cluster_counts = pd.DataFrame(X[feature])
    cluster_counts['Cluster'] = kmeans_clusters
    cluster_counts = cluster_counts.groupby(['Cluster', feature]).size().unstack(fill_value=0)
    cluster_counts = cluster_counts.apply(lambda x: x / x.sum(), axis=1)  # Normalize frequencies

# Univariate Analysis - Histograms
plt.figure(figsize=(16, 8))
for i, feature in enumerate(features, start=1):
    plt.subplot(2, 5, i)
    sns.histplot(data[feature], bins=20, kde=True, color='skyblue')
    plt.title(f"Histogram of {feature}")
plt.tight_layout()
plt.show()

# Bivariate Analysis - Scatter Plots
plt.figure(figsize=(16, 8))
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if i < j:
            plt.subplot(3, 4, (i * 4) + j)
            sns.scatterplot(x=feature1, y=feature2, data=data, hue=kmeans_clusters, palette='viridis')
            plt.title(f"{feature1} vs {feature2}")
plt.tight_layout()
plt.show()

# Multivariate Analysis - Pairplot
sns.pairplot(data, vars=features, hue=kmeans_clusters, palette='viridis')
plt.suptitle("Pairplot of Airlines Data with K-means Clusters", y=1.02)
plt.show()
