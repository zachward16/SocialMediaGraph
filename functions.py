import networkx as nx
import matplotlib.pyplot as plt
import scipy as sci
import numpy as np
import numpy.linalg as linalg
import numpy.random as rand
import pandas as pd
from platform import node
from cdlib.classes.node_clustering import NodeClustering
import community.community_louvain as community_louvain
from community import community_louvain
import cdlib as cd
from collections import defaultdict
import networkx as nx
import cdlib.algorithms as algo
import cdlib.viz as viz
import matplotlib.pyplot as plt

def louvain(
    g_original: object,
    weight: str = "weight",
    resolution: float = 1.0,
    randomize: int = None) -> NodeClustering:
    """
    Louvain  maximizes a modularity score for each community.
    The algorithm optimises the modularity in two elementary phases:
    (1) local moving of nodes;
    (2) aggregation of the network.
    In the local moving phase, individual nodes are moved to the community that yields the largest increase in the quality function.
    In the aggregation phase, an aggregate network is created based on the partition obtained in the local moving phase.
    Each community in this partition becomes a node in the aggregate network. The two phases are repeated until the quality function cannot be increased further.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weight: str, optional the key in graph to use as weight. Default to 'weight'
    :param resolution: double, optional  Will change the size of the communities, default to 1.
    :param randomize: int, RandomState instance or None, optional (default=None). If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.louvain(G, weight='weight', resolution=1.)

    :References:

    Blondel, Vincent D., et al. `Fast unfolding of communities in large networks. <https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta/>`_ Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    .. note:: Reference implementation: https://github.com/taynaud/python-louvain
    """
    g = g_original
    coms = community_louvain.best_partition(
        g, weight=weight, resolution=resolution, randomize=randomize
    )

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [list(c) for c in coms_to_node.values()]
    return cd.NodeClustering(
        coms_louvain,
        g_original,
        "Louvain",
        method_parameters={
            "weight": weight,
            "resolution": resolution,
            "randomize": randomize,
        },
    )
)