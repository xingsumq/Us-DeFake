import torch.nn
from torch import nn
import torch.nn.functional as F
import numpy as np
from utility.utils import *
import embedding.layers as layers


class UnGraphSAINT(nn.Module):
    def __init__(self, arch_gcn, train_params, feat_full, cpu_eval=False):
        super(UnGraphSAINT,self).__init__()
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.out_dim = arch_gcn['dim']
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)
        self.opt_op = None

        # build the model below
        self.num_params = 0
        self.aggregators, num_param = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.out_dim,\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        self.num_params += self.classifier.num_param
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1]
            for l in range(len(dims) - 1)
        ]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])


    def forward(self, node_subgraph, adj_subgraph):
        feat_subg = self.feat_full[node_subgraph] # 读取子图节点属性
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg)) # 处理子图属性
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm))[1] #默认调用了layer中的HighOrderAggregator
        return pred_subg

    def decoder(self, pred_subg):
        # pred_subg = pred_subg.to_sparse()
        adj_rec = torch.sigmoid(torch.matmul(pred_subg.cpu(), pred_subg.cpu().t()))  # 解码器点乘还原邻接矩阵A'
        return adj_rec

    def rec_loss(self, adj_subg, pred_subg, norm_loss):
        rec_subg = self.decoder(pred_subg)
        mse = torch.nn.MSELoss()
        _rl = mse(adj_subg.cpu().to_dense(), rec_subg.cpu())

        return (norm_loss*_rl).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(
                    *self.dims_weight[l],
                    dropout=self.dropout,
                    act=self.act_layer[l],
                    order=self.order_layer[l],
                    aggr=self.aggr_layer[l],
                    bias=self.bias_layer[l],
                    mulhead=self.mulhead,
            )
            num_param += aggr.num_param
            aggregators.append(aggr)
        return aggregators, num_param


    def predict(self, preds): #对embedding按行进行softmax,使得每行所有数值加起来等于1，以进行分类
        return F.softmax(preds, dim=1)


