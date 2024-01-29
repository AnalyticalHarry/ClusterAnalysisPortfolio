import numpy as np
from numpy.linalg import eig

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
      
# # 1. fit Model
# pca = PrincipalComponentAnalysis()
# pcapca.fit(X)

# # 2. covariance matrix
# cov_matrix = pca.get_covariance()
# print(cov_matrix)

# # 3. explained variance ratio
# explained_variance_ratio = pca.get_explained_variance_ratio()
# print(explained_variance_ratio)

# # 4. transformation with a specific number of components
# n = 3
# X_new = pca.transform(X, n)

# # 5. eigenvectors
# print(pca.eigenvectors)

# # 6. eigenvalues
# print(pca.eigenvalues)

### Code created on 29.01.2024
### Hemant Thapa
### hemantthapa1998@gmail.com
