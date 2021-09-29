import numpy as np
import igraph as ig
from numpy.linalg import norm
import math
import random
from collections import Counter

# label used for representing noise
NOISE_LABEL = 'noise'


class SF_SOINN(object):
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
    max_edge_age : int
        Maximum edge age after which an edge gets deleted.
    iter_lambda : int (default: 100)
        Every iter_lambda iterations the grouping process and the node deletion process start.
    pull_factor : int (default: 100)
        Pull factor for node merging.

    Attributes
    ----------
    t : int
        Iteration counter of network update. Increments every time an input signal is processed.
    pull_factor : int (default: 100)
        Pull factor for node merging.
    network : igraph
        The igraph graph representing the incremental neural network.
        Nodes have the following attributes:
            'w' : array, shape = [n_features]
                Weights of the node.
            'wt' : int
                Winning time of the node, i.e., how often that node was selected as the winner node.
            'st' : float
                Similarity threshold of the node.
            'it' : int
                Idle time of the node, i.e., counter of how many iterations the node was not selected as the winner.
            'u' : float
                Utility of a node.
            'c' : string
                Definitive class label of the node.
            'cl' : list
                List of class labels that are assigned to that node.
        Edges have the following attributes:
            'it' : int
                Edge's idle time.
            'wt' : int
                Number of times the edge was reset.
    """

    def __init__(self, x1, x2, x3, max_edge_age, iter_lambda=100, pull_factor=100):
        self.t = 3
        self.max_edge_age = max_edge_age
        self.iter_lambda = iter_lambda
        self.pull_factor = pull_factor

        self.n_del_edges = 0
        self.n_del_nodes = 0

        # generating the graph and adding 3 random training samples
        self.network = ig.Graph()
        self.network.add_vertices(3)
        self.network.vs[0]['w'] = x1
        self.network.vs[1]['w'] = x2
        self.network.vs[2]['w'] = x3
        self.network.vs[0]['wt'] = 1
        self.network.vs[1]['wt'] = 1
        self.network.vs[2]['wt'] = 1
        self.network.vs[0]['it'] = 1
        self.network.vs[1]['it'] = 1
        self.network.vs[2]['it'] = 1
        self.network.vs[0]['cl'] = [NOISE_LABEL]
        self.network.vs[1]['cl'] = [NOISE_LABEL]
        self.network.vs[2]['cl'] = [NOISE_LABEL]
        self.network.vs[0]['c'] = NOISE_LABEL
        self.network.vs[1]['c'] = NOISE_LABEL
        self.network.vs[2]['c'] = NOISE_LABEL

    def _distance(self, a, b):
        """
        Computes the fractional distance between two arrays a and b.

        Parameters
        ----------
        a : array, shape = [n_features]
            First array.
        b : array, shape = [n_features]
            Second array.

        Returns
        -------
        distance : float
            The fractional distance between the two input arrays.
        """
        f = 0.5
        diff = abs(a - b) ** f
        sum = np.sum(diff)
        distance = math.pow(sum, 1/f)
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
        new_node['cl'] = [y]
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
        else:
            # if edge between winner and second winner exists, then reset its idle time
            self.network.es[self.network.get_eid(n1.index, n2.index)]['it'] = 0

        # increment lifetime of all edges that are connected to winner
        for e in self.network.vs[n1.index].incident():
            e['it'] += 1

    def _edge_deletion(self):
        """
        Remove edges that exceed maximum age or connect different clusters.
        """
        for e in self.network.es:
            source = self.network.vs[e.source]
            target = self.network.vs[e.target]
            if e['it'] > self.max_edge_age or source['c'] != target['c']:
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

    def _group(self):
        """
        Determine class labels by selecting the most frequently assigned class for each node.
        """
        # assigning the most frequent label as class label
        for n in self.network.vs:
            occurence_count = Counter(n['cl'])
            n['c'] = occurence_count.most_common(1)[0][0]
            n['cl'] = [n['c']]

    def input_signal(self, x, y=None, learning=True):
        """
        Fit the input signal x. If the input label is not set, then the new input is added with the noise label.

        Parameters
        ----------
        x : array, shape = [n_features]
            The input signal weight vector.
        y : string (default: None)
            The class (label) associated to the input signal.
        learning : boolean (default: True)
            Set to True if learning, False if just prediction. Requires y to be setted.

        Returns
        -------
        prediction : string
            The predicted label. None if prediction fails.
        confidence : float
            The propability that the result is a true positive or true negative.
        """
        n1, n2 = self._get_n1_n2(x)
        if learning:
            if y is None:
                y = NOISE_LABEL

            self.t += 1
            n_nodes = self.network.vcount()
            n_edges = self.network.ecount()

            prediction = n1['c']

            self._similarity_threshold(x, n1)
            self._similarity_threshold(x, n2)

            d1 = self._distance(x, n1['w'])
            d2 = self._distance(x, n2['w'])
            if d1 >= n1['st'] or d2 >= n2['st']:
                self._add_node(x, y)
            else:
                n1['wt'] += 1
                # noise labels should not accumulate, for active learning
                if y != NOISE_LABEL:
                    n1['cl'].append(y)

                self._merge_nodes(n1, x)
                self._linking(n1, n2)

            if self.t % self.iter_lambda == 0:
                if n_nodes > 3:
                    self._nodes_deletion()
                self._group()
                if n_edges > 3:
                    self._edge_deletion()
        else:
            # make prediction, retrieve closest node and output result
            prediction = n1['c']

        # compute confidence of result
        confidence = 0
        if n1 in self.network.vs:
            self._similarity_threshold(x, n1)
            if n1['st'] != None and n1['st'] != 0:
                confidence = 1 - self._distance(
                    x, n1['w']) / n1['st']

        return prediction, confidence
