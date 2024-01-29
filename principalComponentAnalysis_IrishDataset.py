#loading libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn import datasets

#loading dataset
iris = datasets.load_iris()

#building PCA Algorithms
class PrincipalComponentAnalysis:
    def __init__(self, num_components=None):
        self.num_components = num_components
        self.mean = None
        self.std = None
        self.cov_matrix = None
        self.eigenvectors = None
        self.explained_variance_ratio = None

    def fit(self, X):
        #standardise features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if np.any(self.std == 0):
            raise ValueError("One or more features have zero variance.")
        X_std = (X - self.mean) / self.std
        
        #covariance matrix
        self.cov_matrix = np.cov(X_std.T)
        #eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        #eigenvalues and eigenvectors are real
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        #sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        #variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

    def transform(self, X, n_components=None):
        if n_components is None:
            n_components = self.num_components if self.num_components is not None else X.shape[1]
        if n_components > X.shape[1]:
            raise ValueError("Number of components cannot be greater than the number of features.")
        #standardise data based on mean and std from fitting
        X_std = (X - self.mean) / self.std
        #subset of eigenvectors
        eigenvector_subset = self.eigenvectors[:, :n_components]
        #transform the data
        return X_std @ eigenvector_subset

    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio
    
    def get_covariance(self):
        return self.cov_matrix

#feature and target selection 
X = iris.data
y = iris.target

#training model
pca = PrincipalComponentAnalysis(num_components=2)
pca.fit(X)
X_pca = pca.transform(X)

#original plot and transformed plot
plt.figure(figsize=(10, 4))
colors = ['navy', 'turquoise', 'darkorange']
plt.subplot(1, 2, 1)
for i, color in zip(range(len(np.unique(y))), colors):
    subset = X[y == i]
    plt.scatter(subset[:, 0], subset[:, 1], label=iris.target_names[i], color=color, s=20)
plt.title("Original Iris Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, ls='--', alpha=0.5)
plt.legend()

plt.subplot(1, 2, 2)
for i, color in zip(range(len(np.unique(y))), colors):
    subset = X_pca[y == i]
    plt.scatter(subset[:, 0], subset[:, 1], label=iris.target_names[i], color=color, s=20)
plt.title("Iris Data After PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, ls='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

#plot cummulative variance
cumulative_variance = np.cumsum(pca.get_explained_variance_ratio())
plt.figure(figsize=(8, 4))
plt.plot(cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True, ls='--', alpha=0.5)
plt.show()
