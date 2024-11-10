import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance

##Data Generation#######
n_samples = 1500 # Number of samples to generate

############Complete the code in the blank spaces#################
# Set random seed for reproducibility
random_state = 17
######################################

# Generate isotropic Gaussian blobs for clustering
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
# Transformation matrix to introduce anisotropic distribution in data
transformation = [[0.60834549, -0.63667641], [-0.40887718, 0.85253229]]
# Apply the transformation to the data
X_aniso = np.dot(X, transformation)

# Plot initial unlabeled data with anisotropy
plt.figure(figsize=(12,4))
plt.subplot(151)
plt.scatter(X_aniso[:,0], X_aniso[:,1], s=20)
plt.title("Unlabeled data", fontsize=10)

############Complete the code in the blank spaces#################
# KMeans clustering
y_pred = KMeans(n_clusters=3,random_state=random_state).fit_predict(X_aniso) 
y_pred = list(y_pred)

# Plot KMeans clustering result
plt.subplot(152)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred, s=20) # Plot KMeans clustering result
plt.title("KMeans", fontsize=10)

# Gaussian Mixture Model clustering
y_pred_GMM = GaussianMixture(n_components=3, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM = list(y_pred_GMM)

# Plot GMM clustering result
plt.subplot(153)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM, s=20) # Plot GMM clustering result
plt.title("GMM", fontsize=10)

# Calculate RBF kernel matrix for spectral clustering
rbf_param = 7.6
K = np.exp(-distance.cdist(X_aniso, X_aniso, 'sqeuclidean') / (2 * rbf_param ** 2))  # Calculate W_ij and form W matrix
# Calculate degree matrix for normalization
D = np.diag(K.sum(axis=1))  #计算相似度矩阵K的对角矩阵D，用于归一化
# Normalize the kernel matrix using the degree matrix
D_inv_sqrt = np.linalg.inv(np.sqrt(D)) #计算D^(-1/2)
M = D_inv_sqrt @ K @ D_inv_sqrt  #计算M=D^(-1/2)KD^(-1/2)

# Perform SVD to prepare for spectral clustering
U, Sigma, _ = np.linalg.svd(M) # 对M矩阵进行奇异值分解后，U是特征向量，Sigma是特征值
Usubset = U[:,0:3] # 选取前三个特征向量
# KMeans clustering on the normalized eigenvectors for spectral clustering
y_pred_sc = KMeans(n_clusters=3,random_state=random_state).fit_predict(Usubset)
y_pred_sc = list(y_pred_sc)

# Plot custom spectral clustering result
plt.subplot(154)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_sc, s=20) # Plot custom spectral clustering result
plt.title("Spectral Clustering", fontsize=10)

# Sklearn's spectral clustering
y_pred_sc_sklearn = SpectralClustering(n_clusters=3, random_state=random_state,gamma=1/(2*rbf_param**2)).fit_predict(X_aniso)
y_pred_sc_sklearn = list(y_pred_sc_sklearn)

plt.subplot(155)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_sc_sklearn, s=20) # Plot Sklearn's spectral clustering result
plt.title("Spectral Clustering (Sklearn)", fontsize=10)

plt.show()

#########Complete the code ############
# Define a function to calculate the Normalized Mutual Information score
# This function evaluates how similar the clustering results are to the true labels
from sklearn.metrics import normalized_mutual_info_score
def NMI(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

#Print NMI
print("NMI for KMeans:", NMI(y, y_pred))
print("NMI for GMM:", NMI(y, y_pred_GMM))
print("NMI for Spectral Clustering:", NMI(y, y_pred_sc))
print("NMI for Sklearn's Spectral Clustering:", NMI(y, y_pred_sc_sklearn))
