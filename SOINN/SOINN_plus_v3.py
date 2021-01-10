import numpy as np
import igraph as ig
from numpy.linalg import norm
import math
import random


# label used for representing noise
NOISE_LABEL = 'noise'


class SOINN_plus(object):
    """
    Improved version of the Self-organizing incremental neural network called SOINN+.

    Parameters
    ----------
    x1 : array, shape = [n_features]
        First random initialization example.
    x2 : array, shape = [n_features]
        Second random initialization example.
    x3 : array, shape = [n_features]
        Third random initialization example.
    iter_edge_del : int (default: 100)
        Edge removal process starts every iter_edge_del iterations.
    pull_factor : int (default: 100)
        Pull factor for node merging.

    Attributes
    ----------
    t : int
        Iteration counter of network update. Increments every time an input signal is processed.
    pull_factor : int (default: 100)
        Pull factor for node merging.
    b : float (default: 1.4826)
        Scaling factor for the scaled median absolute deviation (sMAD) used in thresholds for node deletion.
    network : igraph
        The igraph graph representing the incremental neural network.
        Nodes have the following attributes:
            'w' : array, shape = [n_features]
                Weights of the node.
            'wt' : int
                Winning time of the node, i.e., how often that node was selected as the winner node.
            'st' : float
                Similarity threshold of the node.
            'T' : float
                Trustworthiness of the node, i.e., ratio of winning time of the node and maximum winning time.
            'it' : int
                Idle time of the node, i.e., counter of how many iterations the node was not selected as the winner.
            'u' : float
                Utility of a node.
            'c' : string
                Class label of the node.
        Edges have the following attributes:
            'it' : int
                Edge's idle time.
            'wt' : int
                Number of times the edge was reset.
            'u' : float
                The utility of the edge.
    """

    def __init__(self, x1, x2, x3, iter_edge_del=100, pull_factor=100):
        self.t = 3
        self.iter_edge_del = iter_edge_del
        self.pull_factor = pull_factor

        self.n_del_edges = 0
        self.n_del_nodes = 0

        # generating the graph and adding 3 random training samples
        self.network = ig.Graph()
        self.network.add_vertices(3)
        self.network.vs[0]['w'] = x1
        self.network.vs[0]['wt'] = 1
        self.network.vs[0]['it'] = 1
        self.network.vs[0]['c'] = NOISE_LABEL
        self.network.vs[1]['w'] = x2
        self.network.vs[1]['wt'] = 1
        self.network.vs[1]['it'] = 1
        self.network.vs[1]['c'] = NOISE_LABEL
        self.network.vs[2]['w'] = x3
        self.network.vs[2]['wt'] = 1
        self.network.vs[2]['it'] = 1
        self.network.vs[2]['c'] = NOISE_LABEL

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
        distance = norm(a - b) ** 2

        return distance

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
        distances = []
        # node is singleton, threshold will be the distance to closest node
        if node.degree() == 0:
            for n in self.network.vs:
                if n.index != node.index:
                    distances.append(self._distance(node['w'], n['w']))
            d = min(distances)
        # node has neighbors, threshold is the distance to furthest neighbor
        else:
            for n in node.neighbors():
                distances.append(self._distance(node['w'], n['w']))
            d = max(distances)
        # threshold depends on the number of winning times, the larger this value the larger the threshold becomes
        if d != 0:
            d += d * (1 - 1 / node['wt'])
            #d += d * (1 - 1 / self.t)
        node['st'] = d

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
        new_node['it'] = 1
        new_node['c'] = y

    def _merge_nodes(self, n1, x):
        """
        Merge winner to the new input signal x so that the winner and its neighbors are adjusted towards the new case x.
        This shift is proportional to the wt of the winning node, the larger wt the less influence has x on the rest of the net.

        Parameters
        ----------
        n1 : igraph vertex
            The winning node.
        x : array, shape = [n_features]
            The weight array of the new node.
        """
        # weights update winner
        n1['w'] = n1['w'] + ((x - n1['w']) / n1['wt'])

        # weights update winner's neighbors
        for n in n1.neighbors():
            pulled_weight = self.pull_factor * n['wt']
            n['w'] = n['w'] + ((x - n['w']) / pulled_weight)

        # winner's idle time is reset
        n1['it'] = 1

    def _linking(self, n1, n2):
        """
        SOINN+ tries to link nodes that are likely to represent signal and not noise.
        Linking depends on the trusworthiness of a node.

        Parameters
        ----------
        n1 : igraph vertex
            The winner node.
        n2 : igraph vertex
            The second winner node.
        """
        # create edge between winner and second winner if it not exists
        n_edges = self.network.ecount()
        if n_edges == 0 or not self.network.are_connected(n1.index, n2.index):
            edge = self.network.add_edge(source=n1.index, target=n2.index)
            edge['it'] = 1
            edge['wt'] = 1

        # if edge between winner and second winner exists, then reset its lifetime
        if self.network.are_connected(n1.index, n2.index):
            self.network.es[self.network.get_eid(n1.index, n2.index)]['it'] = 1
            self.network.es[self.network.get_eid(
                n1.index, n2.index)]['wt'] += 1

        # increment lifetime of all edges that are connected to winner
        for e in self.network.vs[n1.index].incident():
            e['it'] += 1

    def _edge_deletion(self):
        """
        Edge removal algorithm.
        Edges between different clusters should be removed. The assumption behind is that edges between different clusters have a high lifetime.
        If the lifetime of the edge is above the mean, then it will be considered for removal.
        """
        # mean lifetime
        its = []
        for e in self.network.es:
            its.append(e['it'])
        it_mean = np.mean(its)

        # computing edge utilities
        us = []
        max_u = 0.0
        for e in self.network.es:
            e['u'] = e['wt'] / e['it']
            us.append(e['u'])
            if e['u'] > max_u:
                max_u = e['u']
        u_mean = np.mean(us)

        # edge deletion
        if (max_u > 0) and (self.t % self.iter_edge_del == 0):
            for e in self.network.es:
                if e['it'] > it_mean and e['u'] < u_mean:
                    prob_survival = e['u'] / max_u
                    prob_deletion = 1 - prob_survival
                    if prob_deletion > prob_survival:
                        self.network.delete_edges(e.index)
                        self.n_del_edges += 1

    def _nodes_deletion(self):
        """
        Nodes deletion algorithm.
        Nodes are deleted based on their un-utility. High values of unutility mean that the node was often not selected as winner.
        """
        max_u = 0.0
        us = []
        # computing utilities for all nodes
        for n in self.network.vs:
            n['u'] = n['wt'] / n['it']
            us.append(n['u'])
            if n['u'] > max_u:
                max_u = n['u']
        u_mean = np.mean(us)

        # nodes deletion
        for n in self.network.vs:
            if n.degree() == 0 and n['u'] < u_mean:
                prob_survival = n['u'] / max_u
                prob_deletion = 1 - prob_survival
                if prob_deletion > prob_survival:
                    self.network.delete_vertices(n.index)
                    self.n_del_nodes += 1

        # update idle times
        for n in self.network.vs:
            n['it'] += 1

    def fit_input_signal(self, x, y):
        """
        Fit the input signal x.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal weight vector.
        y : string
            The class (label) associated to the input signal.

        Returns
        -------
        prediction : int
            The predicted label.
        """
        self.t += 1
        n_nodes = self.network.vcount()
        n_edges = self.network.ecount()

        n1, n2 = self._get_n1_n2(x)

        self._similarity_threshold(x, n1)
        self._similarity_threshold(x, n2)

        d1 = self._distance(x, n1['w'])
        d2 = self._distance(x, n2['w'])
        if d1 >= n1['st'] or d2 >= n2['st']:
            self._add_node(x, y)
        else:
            n1['c'] = y
            self._merge_nodes(n1, x)
            self._linking(n1, n2)
            if n_edges > 3:
                self._edge_deletion()

        if n_nodes > 3:
            self._nodes_deletion()

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
        n1, _ = self._get_n1_n2(x)
        prediction = n1['c']

        return prediction
