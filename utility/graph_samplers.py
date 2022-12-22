from utility.globals import *
import numpy as np
import scipy.sparse
import graphsaint.cython_sampler as cy


class GraphSampler:
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        pass

    def par_sample(self, stage, **kwargs):
        return self.cy_sampler.par_sample()


class rw_sampling(GraphSampler):
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.RW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.size_root,
            self.size_depth
        )

    def preproc(self, **kwargs):
        pass


class edge_sampling(GraphSampler):
    def __init__(self,adj_train,node_train,num_edges_subgraph):
        """
        The sampler picks edges from the training graph independently, following
        a pre-computed edge probability distribution. i.e.,
            p_{u,v} \\propto 1 / deg_u + 1 / deg_v
        Such prob. dist. is derived to minimize the variance of the minibatch
        estimator (see Thm 3.2 of the GraphSAINT paper).
        """
        self.num_edges_subgraph = num_edges_subgraph
        # num subgraph nodes may not be num_edges_subgraph * 2 in many cases,
        # but it is not too important to have an accurate estimation of subgraph
        # size. So it's probably just fine to use this number.
        self.size_subgraph = num_edges_subgraph * 2
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1 / self.deg_train, 0), shape=adj_train.shape).dot(adj_train)
        super().__init__(adj_train, node_train, self.size_subgraph, {})
        self.cy_sampler = cy.Edge2(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.edge_prob_tri.row,
            self.edge_prob_tri.col,
            self.edge_prob_tri.data.cumsum(),
            self.num_edges_subgraph,
        )

    def preproc(self,**kwargs):
        """
        Compute the edge probability distribution p_{u,v}.
        """
        self.edge_prob = scipy.sparse.csr_matrix(
            (
                np.zeros(self.adj_train.size),
                self.adj_train.indices,
                self.adj_train.indptr
            ),
            shape=self.adj_train.shape,
        )
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}
        self.edge_prob.data *= 2 * self.num_edges_subgraph / self.edge_prob.data.sum()
        # now edge_prob is a symmetric matrix, we only keep the
        # upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format

