# Mining User-aware Multi-Relations for Fake News Detection in Large Scale Online Social Networks

This repository is the official PyTorch implementation of Us-DeFake in the paper:

Mining User-aware Multi-relations for Fake News Detection in Large Scale Online Social Networks, accepted by [*the the 16th ACM International Conference on Web Search and Data Mining*](https://www.wsdm-conference.org/2023/program/accepted-papers) (WSDM '23) [[arXiv](https://arxiv.org/pdf/2212.10778.pdf)].


## Dependencies

* python >= 3.6.8
* pytorch >= 1.1.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* g++ >= 5.4.0
* openmp >= 4.0


## Datasets

To show the input formats of datasets, we give an example dataset "toy" in /data/ directory. The toy dataset is just used to show the input format, it's not suitable for experiments. The structure of the /data/toy/ directory should be as follows.[Download Example Dataset](https://drive.google.com/drive/folders/18IwOQ7hc0S6QaOQxdp7AIHhZezzMZ0CU?usp=sharing)

```
data/
│
└───toy/
    │   class_map.json
    │   post_graph.txt
    │   text_adj_full.npz
    │   text_feats.npy
    │   user_adj_full.npz
    │   user_feats.npy
    │   user_graph
    │   user_post_graph.txt
    └───1/
        │    text_adj_train.npz
        │    text_role.json
        │    user_adj_train.npz
        └─── user_role.json
```
* `class_map.json`: a dictionary of length N. Each key is a node index, and each value is 0 (real news) or 1 (fake news).
* `post_graph.txt`: propagation graph of news. 
* `text_adj_full.npz`: a sparse matrix in CSR format of `post_graph.txt`, stored as a `scipy.sparse.csr_matrix`. The shape is N by N. Non-zeros in the matrix correspond to all the edges in the full graph. It doesn't matter if the two nodes connected by an edge are training, validation or test nodes. 
* `text_feats.npy`: attributes of news.
* `user_adj_full.npz`: a sparse matrix in CSR format of `user_graph.txt`, stored as a `scipy.sparse.csr_matrix`. The shape is M by M. Non-zeros in the matrix correspond to all the edges in the full graph. It doesn't matter if the two nodes connected by an edge are training, validation or test nodes.
* `user_feats.npy`: attributes of users.
* `user_graph.txt`: interaction graph of users.
* `user_post_graph.txt`: posting graph of news and users.  
* `1`: 1st fold of k-fold cross validation. 
* `text_adj_train.npz`: a sparse matrix in CSR format of training news, stored as a `scipy.sparse.csr_matrix`. The shape is also N by N. However, non-zeros in the matrix only correspond to edges connecting two training nodes. The graph sampler only picks nodes/edges from this `text_adj_train`, not `text_adj_full`. Therefore, neither the attribute information nor the structural information are revealed during training. Also, note that only aN rows and cols of `text_adj_train` contains non-zeros. For unweighted graph, the non-zeros are all 1.
* `text_role.json`: a dictionary of four keys. Key `'tr'` corresponds to the list of all training node indices. Key `'va'` corresponds to the list of all validation node indices. Key `'te'` corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. Key `'source news'` corresponds to the source news. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.
* `user_adj_train.npz`: a sparse matrix in CSR format of training users, stored as a `scipy.sparse.csr_matrix`. The shape is also M by M. However, non-zeros in the matrix only correspond to edges connecting two training nodes. The graph sampler only picks nodes/edges from this `user_adj_train`, not `user_adj_full`. Therefore, neither the attribute information nor the structural information are revealed during training. Also, note that only aN rows and cols of `user_adj_train` contains non-zeros. For unweighted graph, the non-zeros are all 1.
* `user_role.json`: a dictionary of four keys. Key `'tr'` corresponds to the list of all training node indices. Key `'va'` corresponds to the list of all validation node indices. Key `'te'` corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. Key `'source news'` corresponds to the source news. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.



## Cython Implemented Parallel Graph Sampler

We have a cython module which need compilation before training can start. Compile the module by running the following from the root directory:

`python graphsaint/setup.py build_ext --inplace`


## Training Configuration

The hyperparameters needed in training can be set via the configuration file: `./train_config/<dataset_name>.yml`.


## Run Training

First of all, please compile cython samplers (see above). 
We suggest looking through the available command line arguments defined in `./utility/globals.py`. 

To run the code on CPU

```
python -m train --data_prefix ./data/<dataset_name> --fold <fold_k> --train_config ./train_config/<dataset_name>.yml --gpu -1
```


To run the code on GPU

```
python -m train --data_prefix ./data/<dataset_name> --fold <fold_k> --train_config ./train_config/<dataset_name>.yml --gpu 0
```

For example, to run dataset 'toy' on CPU:
```
python -m train --data_prefix ./data/toy --fold 1 --train_config ./train_config/toy.yml --gpu -1
```


## Citation & Acknowledgement

We thank Hanqing Zeng et al. proposed the GraphSAINT [paper](https://arxiv.org/abs/1907.04931) and released the [code](https://github.com/GraphSAINT/GraphSAINT). Us-DeFake employs GraphSAINT to learn representations of news and users in large scale online social networks. 

If you find this method helpful for your research, please cite our paper. 


