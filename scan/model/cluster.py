"""
Module for performing agglomerative clustering on fMRI data
"""

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from scan.io.write import ClusterResults


class ClusterAgglomerative:
    """
    Agglomerative clustering model for clustering fMRI data using Ward's method
    and a connectivity matrix formed from a KNN graph.
    """

    def __init__(
        self, 
        n_clusters: int,
        n_neighbors: int = 15,
        metric: str = 'correlation',
        linkage: str = 'ward',
        n_jobs: int = 1,
    ):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.linkage = linkage
        self.n_jobs = n_jobs

    def cluster(self, X: np.ndarray):
        """
        Perform agglomerative clustering.
        """
        # create connectivity matrix
        connectivity = kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors,
            mode='connectivity',
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

        # fit agglomerative clustering model
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            connectivity=connectivity,
            linkage=self.linkage,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_

        return ClusterResults(self.labels_, self.model.get_params())
