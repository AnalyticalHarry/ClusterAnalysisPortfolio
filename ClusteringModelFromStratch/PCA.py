##### Building PCA from Scratch ##### 

# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


class PrincipalComponentAnalysis:
    def __init__(self, num_components=None):
        # constructor method for PCA class
        self.num_components = num_components
        # mean of the data
        self.mean = None  
        # standard deviation of the data
        self.std = None   
        # covariance matrix
        self.cov_matrix = None  
        # eigenvectors
        self.eigenvectors = None  
        # explained variance ratio
        self.explained_variance_ratio = None  

    def fit(self, X):
        # fit method to compute PCA
        # standardise features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if np.any(self.std == 0):
            raise ValueError("One or more features have zero variance.")
        X_std = (X - self.mean) / self.std
        
        # compute covariance matrix
        self.cov_matrix = np.cov(X_std.T)
        
        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        
        # ensure eigenvalues and eigenvectors are real
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        
        # compute explained variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

    def transform(self, X, n_components=None):
        # transform method to reduce dimensionality
        if n_components is None:
            n_components = self.num_components if self.num_components is not None else X.shape[1]
        if n_components > X.shape[1]:
            raise ValueError("Number of components cannot be greater than the number of features.")
        # standardise data based on mean and std from fitting
        X_std = (X - self.mean) / self.std
        # subset of eigenvectors
        eigenvector_subset = self.eigenvectors[:, :n_components]
        # transform the data
        return X_std @ eigenvector_subset

    def get_explained_variance_ratio(self):
        # getter method to return explained variance ratio
        return self.explained_variance_ratio
    
    def get_covariance(self):
        # getter method to return covariance matrix
        return self.cov_matrix

# generate a random dataset
np.random.seed(42)
num_samples = 1000
num_features = 4
X = np.random.randn(num_samples, num_features)

# PCA
pca = PrincipalComponentAnalysis()
pca.fit(X)

# covariance matrix
cov_matrix = pca.get_covariance()
print("Covariance Matrix:")
print(cov_matrix)

# explained variance ratio
explained_variance_ratio = pca.get_explained_variance_ratio()
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# transformation with a specific number of components
n_components = 3
X_new = pca.transform(X, n_components)
print("\nTransformed Data (with", n_components, "components):")
print(X_new)

# eigenvectors
print("\nEigenvectors:")
print(pca.eigenvectors)

# eigenvalues
print("\nEigenvalues:")
print(pca.eigenvalues)

# plot transformed data onto the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5)
plt.title('PCA: Transformed Data (First Two Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

## Auther : Heamnt Thapa
## Date   : 12.02.2024
## Topic  : Building Cluster Algorithms from Scratch 
