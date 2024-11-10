import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

############Complete the code in the blank spaces#################
# Set random seed for reproducibility
#random seed
random_state= 17
# Load the Iris dataset
iris_X = load_iris().data          #data
iris_Y = load_iris().target          #Target labels
##################
#Data visualization 
#Apply t-SNE to reduce the dimensionality of the iris dataset for visualization purposes
tsne = TSNE(n_components=2, init='pca', random_state=random_state) #使用 t-SNE 降维到2维以便可视化
X = tsne.fit_transform(iris_X)

# Set up a figure for visualizing different clustering results
plt.figure(figsize=(12,4))
# Plot the t-SNE output as unlabeled data
plt.subplot(161)
plt.scatter(X[:,0], X[:,1], s=10)
plt.title("Unlabeled data", fontsize=8)

# Plot the actual ground truth labels
plt.subplot(162)
iris_Y = list(iris_Y)
plt.scatter(X[:,0], X[:,1], c=iris_Y, s=10)
plt.title("Ground Truth", fontsize=8)

#########Complete the code in the blank spaces############
y_pred = KMeans(n_clusters=3,random_state=random_state).fit_predict(iris_X) # Apply KMeans clustering
y_pred = list(y_pred)

# Plot KMeans clustering result
plt.subplot(163)
plt.scatter(X[:,0], X[:,1], c=y_pred, s=10)
plt.title("KMeans", fontsize=8)
#####################################

#########Complete the code in the blank spaces############
y_pred_GMM = GaussianMixture(n_components=3, random_state=random_state).fit_predict(iris_X) # Apply Gaussian Mixture Model clustering
y_pred_GMM = list(y_pred_GMM)

# Plot GMM clustering result
plt.subplot(164)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM, s=10)
plt.title("GMM", fontsize=8)


#Calculate RBF kernel matrix
rbf_param = 2
K = np.exp(-distance.cdist(iris_X, iris_X, 'sqeuclidean') / (2 * rbf_param ** 2))
# Compute the degree matrix D
D = np.diag(K.sum(axis=1))
# Normalize the kernel matrix
D_inv_sqrt = np.linalg.inv(np.sqrt(D)) #计算D^(-1/2)
M = D_inv_sqrt @ K @ D_inv_sqrt  #计算M=D^(-1/2)KD^(-1/2)

# Perform SVD to get spectral clustering input
U, Sigma, _ = np.linalg.svd(M)
Usubset = U[:,0:3]
# Apply KMeans clustering on the normalized eigenvectors
y_pred_sc = KMeans(n_clusters=3,random_state=random_state).fit_predict(Usubset)
y_pred_sc = list(y_pred_sc)


plt.subplot(165)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc, s=10) # Plot Spectral Clustering result using custom kernel-based method
plt.title("Spectral Clustering", fontsize=8)

# Apply Spectral Clustering using Sklearn
y_pred_sc_sklearn = SpectralClustering(n_clusters=3, random_state=random_state,gamma=1/(2*rbf_param**2)).fit_predict(iris_X)
y_pred_sc_sklearn = list(y_pred_sc_sklearn)

plt.subplot(166)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc_sklearn, s=10) # Plot Sklearn's spectral clustering result
plt.title("Spectral Clustering (Sklearn)", fontsize=8)

plt.show()

#########Complete the code ############
# Define a function to calculate the Normalized Mutual Information score
# This function evaluates how similar the clustering results are to the true labels
from sklearn.metrics import normalized_mutual_info_score
def NMI(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)

#Print NMI
print("NMI for KMeans:", NMI(iris_Y, y_pred))
print("NMI for GMM:", NMI(iris_Y, y_pred_GMM))
print("NMI for Spectral Clustering:", NMI(iris_Y, y_pred_sc))
print("NMI for Sklearn's Spectral Clustering:", NMI(iris_Y, y_pred_sc_sklearn))

