
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


##################################################
# Question a
##################################################

# location of file
filename = 'C:/Users/fabio/Desktop/Online Retail.csv'

# Load the data with limited rows for faster processing
dataset = read_csv(filename, encoding='windows-1254')

# Display the first few rows to understand the dataset
print(dataset.head(), "\n")


# Data preprocessing: adding a TotalPrice column
dataset['TotalPrice'] = dataset['Quantity'] * dataset['UnitPrice']

# Aggregating the data by CustomerID to get the total quantity and total price for each customer
customer_data = dataset.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

# Feature scaling
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(customer_data[['Quantity', 'TotalPrice']])


# Function to calculate silhouette scores for k-means with clusters between 2 and 8
silhouette_scores = []
range_n_clusters = range(2, 9)

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_dataset)
    silhouette_avg = silhouette_score(scaled_dataset, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
# Silhouette scores for k-means with clusters between 2 and 8
print("Silhouette_scores:", silhouette_scores)


# Plotting the silhouette scores
plt.figure(figsize=(6, 4))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for KMeans Clustering', fontsize=14)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True)
plt.show()

# Determine the 2 best values for the number of clusters
best_clusters = np.argsort(silhouette_scores)[-2:] + 2  # Adding 2 because of 0-indexing
print("The best two clusters are", best_clusters)

##################################################
# Question b
##################################################

# Function for plotting the centroids for the clusters
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)
    
# Function for plotting the Voronoi diagram for the clusters
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# Function for plotting the best two clusters
for i in best_clusters:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit_predict(scaled_dataset)
    kmeans.cluster_centers_
    plt.figure(figsize=(6, 4))
    plot_decision_boundaries(kmeans, scaled_dataset)
    title = str(i) + ' Clusters'
    plt.title(title, fontsize=10)
    plt.show()


    from sklearn.cluster import MiniBatchKMeans, KMeans
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt

# Ensure scaled_dataset is already defined
X = scaled_dataset

k_values = range(2, 10)  # Testing silhouette scores for k from 2 to 9

# Store inertia and time results for both K-Means and Mini-Batch K-Means
times = np.empty((100, 2))   # 100 rows for k = 1 to 100, 2 columns (K-Means, Mini-Batch K-Means)
inertias = np.empty((100, 2))

for k in range(1, 101):  # Running K-means and Mini-batch K-means for k = 1 to 100
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    
    # Measure time taken for K-Means and Mini-Batch K-Means (run once for each)
    times[k-1, 0] = timeit(lambda: kmeans_.fit(scaled_dataset), number=1)
    times[k-1, 1] = timeit(lambda: minibatch_kmeans.fit(scaled_dataset), number=1)
    
    # Store inertia
    inertias[k-1, 0] = kmeans_.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_

# Plotting the graphs for inertia and time comparison
plt.figure(figsize=(10,4))

# Subplot for inertia
plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis([1, 100, 0, max(inertias[:, 0].max(), inertias[:, 1].max())])

# Subplot for time
plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
plt.axis([1, 100, 0, times[:, 1].max()])

plt.show()


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]


# Ensure scaled_dataset is already defined
X = scaled_dataset

# Store inertia results for K-Means only
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

# Plotting inertia vs. k to find the elbow
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(2, inertias[1]),  # Change the coordinates based on your data
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis([1, 9, 0, max(inertias) * 1.1])  # Adjust the y-axis limit dynamically
plt.title("Inertia vs. Number of Clusters (k)", fontsize=16)  # Add a title for clarity
plt.grid()  # Optional: Add grid for better readability
plt.show()

# After finding the elbow, you can use the best k values
best_k_values = [2, 3]  # Example based on your earlier findings

# Store time results for K-Means and Mini-Batch K-Means for best k values
times = np.empty((len(best_k_values), 2))
inertias_best_k = np.empty((len(best_k_values), 2))

for idx, k in enumerate(best_k_values):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    
    # Measure time taken for K-Means and Mini-Batch K-Means
    times[idx, 0] = timeit(lambda: kmeans_.fit(X), number=1)
    times[idx, 1] = timeit(lambda: minibatch_kmeans.fit(X), number=1)
    
    # Store inertia
    inertias_best_k[idx, 0] = kmeans_.inertia_
    inertias_best_k[idx, 1] = minibatch_kmeans.inertia_

# Plotting the graphs for inertia and time comparison
plt.figure(figsize=(10, 4))

# Subplot for inertia
plt.subplot(121)
plt.plot(best_k_values, inertias_best_k[:, 0], "r--", label="K-Means")
plt.plot(best_k_values, inertias_best_k[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Inertia for Best k Values", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(best_k_values)  # Only show best k values on x-axis

# Subplot for time
plt.subplot(122)
plt.plot(best_k_values, times[:, 0], "r--", label="K-Means")
plt.plot(best_k_values, times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training Time for Best k Values", fontsize=14)
plt.xticks(best_k_values)  # Only show best k values on x-axis

plt.show()


k = 2 

# K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_dataset)

# Mini-Batch K-Means
minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
minibatch_labels = minibatch_kmeans.fit_predict(scaled_dataset)



from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import seaborn as sns


# Assuming 'scaled_dataset' contains your scaled features
X = scaled_dataset  # Ensure this contains Quantity and TotalPrice

# Perform clustering
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
minibatch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42).fit(X)

# Evaluate clustering
ch_index_kmeans = calinski_harabasz_score(X, kmeans.labels_)
db_index_kmeans = davies_bouldin_score(X, kmeans.labels_)

ch_index_minibatch = calinski_harabasz_score(X, minibatch_kmeans.labels_)
db_index_minibatch = davies_bouldin_score(X, minibatch_kmeans.labels_)

# Print results
print("Calinski-Harabasz Index for K-Means:", ch_index_kmeans)
print("Davies-Bouldin Index for K-Means:", db_index_kmeans)
print("Calinski-Harabasz Index for Mini-Batch K-Means:", ch_index_minibatch)
print("Davies-Bouldin Index for Mini-Batch K-Means:", db_index_minibatch)

# Correlation matrix
correlation_matrix = pd.DataFrame(X, columns=['Quantity', 'TotalPrice']).corr()
print("Correlation Matrix:\n", correlation_matrix)

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

