import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


PCA_NUM_COMPONENTS = 6
K_MEANS_CLUSTERS = 7
GAUSSIAN_MIXTURE_COMPONENTS = 8

def classify():
	# Reduce dimensionality using PCA
	data = pd.read_csv('../data/complete.csv')
	pca = PCA(n_components=PCA_NUM_COMPONENTS)
	pca.fit(data)
	data_pca = pca.transform(data)
	pdf = pd.DataFrame(data=data_pca, columns=['Component' + str(i) for i in range(1, PCA_NUM_COMPONENTS + 1)])

	# Perform K-MEANS Clustering
	positions1 = kMeansClustering(data, pdf)

	# Perform Gaussian Mixture Model Clustering
	positions2 = gaussianClustering(data, pdf)

	return (positions1, positions2)


def kMeansClustering(true_data, data):
	k = K_MEANS_CLUSTERS
	kmeans = KMeans(n_clusters=k).fit(data)
	positions = kmeans.labels_
	true_data['Position_K'] = positions
	return true_data[['PLAYER_ID', 'Position_K']]

'''
Uses "Elbow" method to find optimal number of clusters for k means clustering
'''
def findOptimalKMeansClustersElbow(data):
	sum_sq_dist = []
	K = range(2, 15)
	for k in K:
		km = KMeans(n_clusters=k)
    	km = km.fit(data)
    	sum_sq_dist.append(km.inertia_)
    plt.plot(K, sum_sq_dist)

def gaussianClustering(true_data, data):
	n = GAUSSIAN_MIXTURE_COMPONENTS
	gmm = GaussianMixture(n_components=n)
	gmm.fit(data)
	labels = gmm.predict(data)
	true_data['Position_G'] = positions
	return true_data[['PLAYER_ID', 'Position_G']]


'''
Uses Bayesian Information Criterion to find optimal number of
components for Gaussian mixture clustering.
'''
def findOptimalGaussianComp(data):
	n_components = np.arange(2, 15)
	models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data) for n in n_components]
	plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
	plt.scatter(n_components, [m.bic(data) for m in models])
	plt.xlabel('Number of Clusters')
	plt.ylabel('BIC')
	plt.title('Bayesian Information Criterion (bic) method')
	plt.show()

'''
Plots the cumulative explained variance based on the number of components used
Target cum. explained variance used: .95
'''
def findOptimalPCANumComp():
	data = pd.read_csv('../data/complete.csv')
	data_rescaled = rescaleData(data)
	pca = PCA().fit(data_rescaled)

	plt.rcParams["figure.figsize"] = (12,6)

	fig, ax = plt.subplots()
	xi = np.arange(1, 19, step=1)
	y = np.cumsum(pca.explained_variance_ratio_)

	plt.ylim(0.0,1.1)
	plt.plot(xi, y, marker='o', linestyle='--', color='b')

	plt.xlabel('Number of Components')
	plt.xticks(np.arange(0, 11, step=1))
	plt.ylabel('Cumulative variance (%)')
	plt.title('The number of components needed to explain variance')

	plt.axhline(y=0.95, color='r', linestyle='-')
	plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

	ax.grid(axis='x')
	plt.show()


def rescaleData(data):
	features = [x for x in data.columns if (x != 'PLAYER_ID') & (x != 'MIN')]
	x = df2.loc[:, features].values
	scaled_x = StandardScaler().fit_transform(x)
	return scaled_x
