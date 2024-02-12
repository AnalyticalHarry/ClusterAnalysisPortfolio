##### Building K-means from Scratch ##### 

#Importing libaries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#creating KMeans
class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        # the number of clusters to form as well as the number of centroids to generate.
        self.k = k
        # the percentage change in the centroids before considering the algorithm has converged.
        self.tolerance = tolerance
        # the maximum number of times the algorithm will attempt to converge before stopping.
        self.max_iterations = max_iterations
        
    # method to fit model inside dataset
    def fit(self, data):
        # initialise Centroids Randomly
        # empty dictonary to store centroids
        self.centroids = {}
        # for loop to iterate inside k
        for i in range(self.k):
            self.centroids[i] = data[i]
        # loop through max iterations times
        for i in range(self.max_iterations):
            # create empty classes for each centroid
            self.classes = {i: [] for i in range(self.k)}
            # assigning data points to nearest centroid
            for features in data:
                # euclidean distance between the current data point and each centroid,
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
            # previous centroids
            previous = dict(self.centroids)
            # update centroids as mean of assigned data points
            for classification in self.classes:
                self.centroids[classification] = np.mean(self.classes[classification], axis=0)
            # convergence
            isOptimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            # break loop if converged
            if isOptimal:
                break
    # method to predict
    def prediction(self, data):
        # euclidean distance
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
def within_cluster_sum_of_squares(data, k):
    kmeans = K_Means(k)
    kmeans.fit(data)
    wcss = sum([np.sum([np.linalg.norm(x - kmeans.centroids[ci])**2 for x in kmeans.classes[ci]]) for ci in kmeans.classes])
    return wcss

# clustering plot
def clustering(km, x1, x2):
    plt.figure(figsize=(10, 6))
    # total colors for plotting clusters
    colors = ["black", "grey", "cyan", "green", "blue", "magenta", "yellow", "white", "orange", "purple", "brown", "pink"]
    
    # iterate over each cluster
    for classification in km.classes:
        color = colors[classification]
        for features in km.classes[classification]:
            # plotting each data point in the cluster
            plt.scatter(features[0], features[1], color=color, s=40, alpha=0.5)
    
    # initialise centroid label counter
    centroid_label = 1
    
    # iterate over each centroid and plot them with number labels
    for centroid in km.centroids:
        # plotting centroid of cluster
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=300, marker="X", color='red')
        # text label at centroid center
        plt.text(km.centroids[centroid][0], km.centroids[centroid][1], str(centroid_label), ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        # increment centroid label counter
        centroid_label += 1
    
    plt.grid(True, ls='--', alpha=0.2, color='grey')
    plt.title("K-Means Clustering on Random Data")
    plt.xlabel(f"{x1}")
    plt.ylabel(f"{x2}")
    plt.show()

# elbow plot
def elbow(data):
    # K values range from 1 to 10
    k_values = range(1, 10)
    # formula for with-in cluster sum of squares
    wcss_scores = [within_cluster_sum_of_squares(data, k) for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_scores, '-o', color='red')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.xticks(k_values)
    plt.grid(True, ls='--', alpha=0.2, color='grey')
    plt.show()

# generating dataset for testing
np.random.seed(42)
# creating data from random numbers
data = np.random.randn(1000, 2)
print(data[:10])

# K_Means for the clustering
km = K_Means(3)
# fitting model
km.fit(data)

# checking centroids
km.centroids

# clustering plot
clustering(km, "feature1", "feature2")

#elbow plot
elbow(data)

## Auther : Heamnt Thapa
## Date   : 12.02.2024
## Topic  : Building Cluster Algorithms from Scratch 
