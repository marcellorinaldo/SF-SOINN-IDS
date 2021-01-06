import numpy as np
import igraph as ig
from numpy.linalg import norm
import math
from collections import Counter


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
    pull_factor : int (default: 100)
        Pull factor for node merging.
    b : float (default: 1.4826)
        Scaling factor for the scaled median absolute deviation (sMAD) used in thresholds for node deletion.

    Attributes
    ----------
    t : int
        Iteration counter of network update. Increments every time an input signal is processed.
    pull_factor : int (default: 100)
        Pull factor for node merging.
    max_wt : int
        Maximum winning time between all nodes.
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
                Unutility of a node.
            'c' : string
                Class label of the node.
        Edges have the following attributes:
            'lt' : int
                Edge's lifetime.
    st_n1_mean : float
        Arithmetic mean of the similarity thresholds of the winners that were linked to second winners.
    st_n2_mean : float
        Arithmetic mean of the similarity thresholds of the second winners that were linked to winners.
    st_n1_sd : float
        Standard deviation of the similarity thresholds of the winners that were liked to second winners.
    st_n2_sd : float
        Standard deviation of the similarity thresholds of the second winners that were liked to winners.
    n_del_nodes: int
        Number of deleted nodes.
    n_del_edges: int
        Number of deleted edges.
    del_nodes_unutility_mean: float
        Mean of un-utilities of deleted nodes.
    """

    def __init__(self, x1, x2, x3, pull_factor=100, b=1.4826):
        self.t = 3
        self.pull_factor = pull_factor
        self.max_wt = 1

        # generating the graph and adding 3 random training samples
        self.network = ig.Graph()
        self.network.add_vertices(3)
        self.network.vs[0]['w'] = x1
        self.network.vs[0]['wt'] = 1
        self.network.vs[0]['it'] = 0
        self.network.vs[0]['c'] = NOISE_LABEL
        self.network.vs[1]['w'] = x2
        self.network.vs[1]['wt'] = 1
        self.network.vs[1]['it'] = 0
        self.network.vs[1]['c'] = NOISE_LABEL
        self.network.vs[2]['w'] = x3
        self.network.vs[2]['wt'] = 1
        self.network.vs[2]['it'] = 0
        self.network.vs[2]['c'] = NOISE_LABEL

        self.st_n1_mean = 0.0
        self.st_n2_mean = 0.0
        self.st_n1_sd = 0.0
        self.st_n2_sd = 0.0

        self.n_del_nodes = 0
        self.n_del_edges = 0
        self.del_nodes_unutility_mean = 0

        self.b = b

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
        new_node['it'] = 0
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
        # update max_wt
        n1['wt'] += 1
        if n1['wt'] > self.max_wt:
            self.max_wt = n1['wt']

        # weights update winner
        n1['w'] = n1['w'] + ((x - n1['w']) / n1['wt'])

        # weights update winner's neighbors
        for n in n1.neighbors():
            pulled_weight = self.pull_factor * n['wt']
            n['w'] = n['w'] + ((x - n['w']) / pulled_weight)

        # winner's idle time is reset
        n1['it'] = 0

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
        # compute trustworthiness of each node
        for n in self.network.vs:
            # used another formula than in paper: removed -1 (unnecessary)
            n['T'] = n['wt'] / (self.max_wt)

        # create an edge between winner and second winner if the 3 conditions hold
        n_edges = self.network.ecount()
        cond1 = n_edges < 3
        cond2 = n1['st'] * (1 - n1['T']) < self.st_n1_mean + 2 * self.st_n1_sd
        cond3 = n2['st'] * (1 - n2['T']) < self.st_n2_mean + 2 * self.st_n2_sd

        if cond1 or cond2 or cond3:
            # create edge between winner and second winner if it not exists
            if n_edges == 0 or self.network.are_connected(n1.index, n2.index) == False:
                edge = self.network.add_edge(source=n1.index, target=n2.index)
                edge['lt'] = 0

                if n_edges < 1:
                    n_edges = 1

                # incrementally update the means and standard deviations with the following equations:
                #       mean_n = mean_n-1 + (x_n - mean_n-1)/n
                #       S_n = S_n-1 + (x_n - mean_n-1)(x_n - mean_n)
                #       sigma_n = sqrt(S_n / n)
                old_n1_mean = self.st_n1_mean
                old_n2_mean = self.st_n2_mean
                self.st_n1_mean = self.st_n1_mean + \
                    (abs(n1['st'] - self.st_n1_mean) / n_edges)
                self.st_n2_mean = self.st_n2_mean + \
                    (abs(n2['st'] - self.st_n2_mean) / n_edges)
                var1 = (self.st_n1_sd + abs(n1['st'] - old_n1_mean)
                        * abs(n1['st'] - self.st_n1_mean)) / n_edges
                var2 = (self.st_n2_sd + abs(n2['st'] - old_n2_mean)
                        * abs(n2['st'] - self.st_n2_mean)) / n_edges
                self.st_n1_sd = math.sqrt(var1)
                self.st_n2_sd = math.sqrt(var2)

        # if edge between winner and second winner exists, then update its lifetime
        if self.network.are_connected(n1.index, n2.index):
            self.network.es[self.network.get_eid(n1.index, n2.index)]['lt'] = 0

        # increment lifetime of all edges that are connected to winner
        for e in self.network.vs[n1.index].incident():
            e['lt'] += 1

    def _edge_deletion(self, n1):
        """
        Edge removal algorithm.
        Edges between different clusters should be removed. This process can left some nodes unconnected, which are later candidates for removal.
        Edge removal is performed if the edge's lifetime is exceptionally high, defining a threshold based on the lifetimes of previously deleted edges.

        Parameters
        ----------
        n1 : igraph vertex
            The winner node.
        """
        if n1.degree() > 0:
            # retrieve the lifetimes of edges that are connected to the winner
            lifetimes = []
            for e in n1.incident():
                lifetimes.append(e['lt'])

            # interquartile range = q3 - q1
            q3, q1 = np.percentile(lifetimes, [75, 25])

            # outlier to the 3rd quartile factor
            outlier_factor = np.percentile(lifetimes, 75) + 2 * (q3 - q1)

            # threshold for edge deletion
            prob = self.n_del_edges / (self.n_del_edges + len(lifetimes))
            threshold_edge_del = np.mean(
                lifetimes) * prob + outlier_factor * (1 - prob)

            # edge deletion
            for n in n1.neighbors():
                edge = self.network.get_eid(n1.index, n.index)
                if self.network.es[edge]['lt'] > threshold_edge_del:
                    self.n_del_edges += 1
                    self.network.delete_edges(edge)

    def _nodes_deletion(self):
        """
        Nodes deletion algorithm.
        Nodes are deleted based on their un-utility. High values of unutility mean that the node was often not selected as winner.
        """
        # computing un-utilities for all nodes and retrieving the unconnected nodes
        unutilities = []
        n_unconnected_nodes = 0
        for n in self.network.vs:
            n['u'] = n['it'] / n['wt']
            if n.degree() < 1:
                n_unconnected_nodes += 1
            else:
                unutilities.append(n['u'])

        n_nodes = self.network.vcount()

        # computing weights and factors
        unutilities_med = np.median(unutilities)
        outlier_factor = unutilities_med + self.b * \
            np.median([(u - unutilities_med) for u in unutilities])
        noise_ratio = n_unconnected_nodes / n_nodes

        if (self.n_del_nodes + n_nodes - n_unconnected_nodes) == 0:
            prob = self.n_del_nodes
        else:
            prob = self.n_del_nodes / \
                (self.n_del_nodes + n_nodes - n_unconnected_nodes)

        # threshold for node deletion
        threshold_node_del = self.del_nodes_unutility_mean * \
            prob + outlier_factor * (1 - prob) * (1 - noise_ratio)

        # node deletion
        n_edges = self.network.ecount()
        for n in self.network.vs:
            if n_edges > 0 and n.degree() == 0 and n['u'] > threshold_node_del:
                self.n_del_nodes += 1
                self.del_nodes_unutility_mean += abs(
                    n['u'] - self.del_nodes_unutility_mean) / self.n_del_nodes
                self.network.delete_vertices(n.index)

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

        n1, n2 = self._get_n1_n2(x)

        self._similarity_threshold(x, n1)
        self._similarity_threshold(x, n2)

        # add new node to the network if one of the conditions hold
        d1 = self._distance(x, n1['w'])
        d2 = self._distance(x, n2['w'])
        if d1 >= n1['st'] or d2 >= n2['st']:
            self._add_node(x, y)
        else:
            n1['c'] = y
            # new node is not added but merged with its first winner
            self._merge_nodes(n1, x)
            # linking nodes depending on trustworthiness
            self._linking(n1, n2)
            # edge deletion
            if self.network.ecount() > 0:
                self._edge_deletion(n1)

        # nodes deletion
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
        n1, n2 = self._get_n1_n2(x)
        prediction = n1['c']

        return prediction
