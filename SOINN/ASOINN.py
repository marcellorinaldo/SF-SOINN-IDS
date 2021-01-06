import numpy as np
import igraph as ig
from numpy.linalg import norm
import math
from collections import Counter


# label used for representing noise
NOISE_LABEL = 'noise'


class ASOINN(object):
    """
    Improved version of the Self-organizing incremental neural network called SOINN+.
    Typical combinations for lambda_iter and max_edge_age are (100, 50), (300, 100), (50, 10), (25, 25).
    Performance will be worse if parameters have high value, and better if parameters have low value.
    Higher values parameters will retain more information and make the network grow larger.

    Parameters
    ----------
    x1 : array, shape = [n_features]
        First random initialization example.
    x2 : array, shape = [n_features]
        Second random initialization example.
    pull_factor : int (default: 100)
        Pull factor for node merging.
    lambda_iter : int
        Lambda parameter for iterations number.
    max_edge_age : int
        Threshold for edge deletion.

    Attributes
    ----------
    t : int
        Iteration counter of network update. Increments every time an input signal is processed.
    network : igraph
        The igraph graph representing the incremental neural network.
        Nodes have the following attributes:
            'w' : array, shape = [n_features]
                Weights of the node.
            'wt' : int
                Winning time of the node, i.e., how often that node was selected as the winner node.
            'st' : float
                Similarity threshold of the node.
            'c' : list
                List of class labels (integers) of the node.
        Edges have the following attributes:
            'lt' : int
                Edge's lifetime.
    """

    def __init__(self, x1, x2, lambda_iter, max_edge_age, pull_factor=100):
        self.t = 2
        self.lambda_iter = lambda_iter
        self.max_edge_age = max_edge_age
        self.pull_factor = pull_factor

        # generating the graph and adding 3 random training samples
        self.network = ig.Graph()
        self.network.add_vertices(2)
        self.network.vs[0]['w'] = x1
        self.network.vs[0]['wt'] = 1
        self.network.vs[0]['c'] = NOISE_LABEL
        self.network.vs[1]['w'] = x2
        self.network.vs[1]['wt'] = 1
        self.network.vs[1]['c'] = NOISE_LABEL

    def fit_input_signal(self, x, y):
        """
        Fit the input signal x.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal weight vector.
        y : string
            The class (label) associated to the input signal.
        """
        self.t += 1

        n1, n2 = self._get_n1_n2(x)

        self._similarity_threshold(x, n1)
        self._similarity_threshold(x, n2)

        # add new node to the network if one of the conditions hold
        d1 = self._distance(x, n1['w'])
        d2 = self._distance(x, n2['w'])
        if d1 > n1['st'] or d2 > n2['st']:
            self._add_node(x, y)
        else:
            n1['wt'] += 1
            n1['c'] = y

            # connect winner and second winner if not connected
            if not self.network.are_connected(n1.index, n2.index):
                edge = self.network.add_edge(source=n1.index, target=n2.index)
                edge['lt'] = 0

            # increment lifetime of all edges that are connected to winner
            for e in self.network.vs[n1.index].incident():
                e['lt'] += 1

            # weights update winner
            n1['w'] = n1['w'] + ((x - n1['w']) / n1['wt'])

            # weights update of winner's neighbors
            for n in n1.neighbors():
                pulled_weight = self.pull_factor * n['wt']
                n['w'] = n['w'] + ((x - n['w']) / pulled_weight)

            # delete edges with a lifetime greater than max_edge_age and remove resulting singleton nodes
            for e in self.network.es:
                if e['lt'] > self.max_edge_age:
                    source = self.network.vs[e.source]
                    target = self.network.vs[e.target]
                    self.network.delete_edges(e)
                    if source.degree() == 0:
                        self.network.delete_vertices(source.index)
                    if target.degree() == 0:
                        self.network.delete_vertices(target.index)

        # if t is multiple of lambda_iter, remove nodes
        if self.t % self.lambda_iter == 0:
            # remove nodes with 1 or less neighbors
            for n in self.network.vs:
                if n.degree() <= 1:
                    self.network.delete_vertices(n.index)

    def predict(self, x):
        """
        Make prediction.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal.

        Returns
        -------
        prediction : string
            The predicted label.
        """
        n1, n2 = self._get_n1_n2(x)
        prediction = n1['c']

        return prediction

    def _distance(self, a, b):
        """
        Computes the squared Euclidean distance between two arrays a and b.

        Parameters
        ----------
        a : array, shape = [n_features]
            First array.
        b : array, shape = [n_features]
            Second array.

        Returns
        -------
        distance : float
            The squared Euclidean distance between the two input arrays.
        """
        return norm(a - b) ** 2

    def _get_n1_n2(self, x):
        """
        Computes winner and second winner.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal to be processed.

        Returns
        -------
        n1 : igraph vertex
            The winner node as an igraph vertex.
        n2 : igraph vertex
            The second winning node as an igraph vertex.
        """
        # determine winner and second winner
        distances = dict()
        for n in self.network.vs:
            distances[n] = self._distance(x, n['w'])
        n1 = min(distances, key=distances.get)
        del distances[n1]
        n2 = min(distances, key=distances.get)

        return n1, n2

    def _similarity_threshold(self, x, node):
        """
        Computes similarity thresholds of winner and second winner.
        If winner (or second winner) is a singleton node, then threshold will correpond to the distance to the closest node.
        If winner (or second winner) is not a singleton, then threshold will correspond to the distance to the furthest neighbor.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal to be processed.
        node : igraph vertex
            The winner (or second winner) node as an igraph vertex.
        """
        # node is singleton, threshold will be the distance to closest node
        if node.degree() == 0:
            distances = []
            for n in self.network.vs:
                if n.index != node.index:
                    distances.append(self._distance(node['w'], n['w']))
            node['st'] = min(distances)
        # node has neighbors, threshold is the distance to furthest neighbor
        else:
            distances = []
            for n in node.neighbors():
                distances.append(self._distance(node['w'], n['w']))
            node['st'] = max(distances)

    def _add_node(self, weights, y):
        """
        Add new node to the network.

        Parameters
        ----------
        weights : array, shape = [n_features]
            The weights of the new node to add.
        y : string
            The class (label) associated to the input signal.
        """
        new_node = self.network.add_vertex()
        new_node['w'] = weights
        new_node['wt'] = 1
        new_node['c'] = y
