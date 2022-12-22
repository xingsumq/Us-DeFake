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

To show the input formats of datasets, we give an example dataset "toy" in /data/ directory. The toy dataset is just used to show the input format, it's not suitable for experiments. The structure of the /data/toy/ directory should be as follows.

```
data/
│
└───toy/
    │   class_map.json
    │   post_graph.txt
    │   text_adj_full.npz
    └───1/
        │    text_adj_train.npz
        │    text_role.json
        │    user_adj_train.npz
        └─── user_role.json
```
* `class_map.json`: 
* `post_graph.txt`:
* `text_adj_full.npz`:
* `1`:
* `text_adj_train.npz`:
* `text_role.json`:
* `user_adj_train.npz`:
* `user_role.json`:



## Cython Implemented Parallel Graph Sampler

We have a cython module which need compilation before training can start. Compile the module by running the following from the root directory:

`python graphsaint/setup.py build_ext --inplace`


## Training Configuration


## Run Training


## Citation & Acknowledgement
