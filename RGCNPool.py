from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
import torch.optim as optim
###from torchvision import datasets, transforms
#from operator import itemgetter 
#from torchvision.transforms import v2
#import random
import torch.masked
from torch_geometric.nn import BatchNorm, RGCNConv

from torch_geometric.utils import coalesce, scatter, softmax, remove_self_loops
import sys


def fix_pooled_edge_attrs(new_edge_attrs):
    for i, e in enumerate(new_edge_attrs):
        if sum(e) > 1:
            #print('in if statement at row', i)
            ind = next(x for x, val in enumerate(e)if val >= 1)
            e = torch.zeros(20)
            e[ind] = 1
            new_edge_attrs[i] = e
    return new_edge_attrs

class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor
class EdgePoolingRGCN(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`__ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ paper, use
    either :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0.0`.

    To duplicate the configuration from the `"Edge Contraction Pooling for
    Graph Neural Networks" <https://arxiv.org/abs/1905.10990>`__ paper,
    set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (callable, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0.0`)
        add_to_edge_score (float, optional): A value to be added to each
            computed edge score. Adding this greatly helps with unpooling
            stability. (default: :obj:`0.5`)
    """
    def __init__(
        self,
        in_channels: int,
        min_node_score: float,
        min_edge_score: float,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.min_node_score = min_node_score
        self.min_edge_score = min_edge_score
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.lin2 = torch.nn.Linear(2 * in_channels, 2*in_channels)
        #self.lin3 = torch.nn.Linear(in_channels, int(in_channels/2))
        self.lin = torch.nn.Linear(2*in_channels, 1)
        self.sig = torch.nn.Sigmoid()
        self.reset_parameters()
        #self.bn1 = torch.nn.BatchNorm1d(in_channels)
        #self.bn2 = torch.nn.BatchNorm1d(int(in_channels/2))


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()


    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        #print(raw_edge_score)
        #print(edge_index)
       # print(softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes))
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)


    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)


    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        #embs: Tensor,
        edge_index: Tensor,
        edge_attr: List[Tensor],
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        #IF GETTING WEIRD ERRORS WITH THIS U ADDED AN EMBS FIELD TO THE FORWARD WHICH DID NOT EXIST IN OTHER NETS
        #e = torch.cat([embs[edge_index[0]], embs[edge_index[1]]], dim=-1)
        edge_score = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_score = F.relu(self.lin2(edge_score))
        #e = F.relu(self.lin3(e))
        #e = F.dropout(e, p=self.dropout, training=self.training)
        edge_score = self.lin(edge_score).view(-1)
        #e = F.dropout(e, p=self.dropout, training=self.training)
        edge_score = self.sig(edge_score)
        #print("edges scores:", edge_score)
        #edge_score = (edge_score-torch.mean(edge_score))/torch.std(edge_score)
        #SM_e = self.compute_edge_score_softmax(e, edge_index, x.size(0))
        #e = e + self.add_to_edge_score
        #print('SOFTMAX TOTAL VALUE', sum(SM_e))
        x, edge_index, edge_attr, batch, unpool_info = self._merge_edges(
            x, edge_index, edge_attr, batch, edge_score)
        #unpool_info=UnpoolInfo(edge_index=unpool_info.edge_index, cluster=unpool_info.cluster,
                                 #batch=unpool_info.batch, new_edge_score=e)
        #print(edge_score)
        #print(edge_score)
        return x, edge_score, edge_attr, edge_index, batch, unpool_info

        #FOR EDGE ATTR CAN JUST ADD THAT IN TO THE COALESCE BUT NEED A LIST OF TENSORS


    def _merge_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        
        cluster = torch.empty_like(batch)
        edge_attr_list = [r for r in edge_attr]
        #print(edge_attr_list[0].shape)
        #print('EDGE INDEX:', edge_index.shape)
        #SM_edge_score = self.compute_edge_score_softmax(edge_score, edge_index, x.size(0)
        perm: List[int] = torch.argsort(edge_score, descending=True, stable=True).tolist()
        #print("perm:", perm)
        #print(perm[:20])
        #good_x_inds = set((t >= self.min_node_score).nonzero(as_tuple=True)[0].tolist())

        
        

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        mask = torch.ones(x.size(0), dtype=torch.bool)
        #print(perm)

        #print(edge_score)
        i = 0
        new_edge_indices: List[int] = []
        edge_index_cpu = edge_index.cpu()
        counter = 0
        for edge_idx in perm:
            counter += 1
            if counter >= len(perm)*(1-self.min_edge_score):
                break

            #if edge_index[0][edge_idx] not in good_x_inds and edge_index[1][edge_idx] not in good_x_inds:
                #continue
            source = int(edge_index_cpu[0, edge_idx])
            if not bool(mask[source]):
                continue

            target = int(edge_index_cpu[1, edge_idx])
            if not bool(mask[target]):
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            mask[source] = False

            if source != target:
                cluster[target] = i
                mask[target] = False

            i += 1

        # The remaining nodes are simply kept:
        j = int(mask.sum())
        cluster[mask] = torch.arange(i, i + j, device=x.device)
        i += j
        #print(cluster)
        # We compute the new features as an addition of the old ones.
        new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')
        new_edge_score = edge_score[new_edge_indices]
        if int(mask.sum()) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            remaining_score = remaining_score*100
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        
        new_edge_index, new_edge_attr = coalesce(cluster[edge_index], edge_attr, num_nodes=new_x.size(0))
        new_edge_attr = fix_pooled_edge_attrs(new_edge_attr)
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch, new_edge_score=new_edge_score)
        #print("unpooling info:\n", unpool_info)
        return new_x, new_edge_index, new_edge_attr, new_batch, unpool_info

    def unpool(
        self,
        x: Tensor,
        unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (torch.Tensor): The node features.
            unpool_info (UnpoolInfo): Information that has been produced by
                :func:`EdgePooling.forward`.

        Return types:
            * **x** *(torch.Tensor)* - The unpooled node features.
            * **edge_index** *(torch.Tensor)* - The new edge indices.
            * **batch** *(torch.Tensor)* - The new batch vector.
        """
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
    
class varRGCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels):
        super(varRGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.end_layer = RGCNConv(in_channels, 1, 20)

        self.num_layers = num_layers
        self.norms = torch.nn.ModuleList()
        self.n1 = nn.BatchNorm1d(in_channels)
        for _ in range(self.num_layers - 1):
            self.layers.append(RGCNConv(in_channels, in_channels, 20))
        self.layers.append(self.end_layer)
        #for conv in convs:
        for _ in range(self.num_layers -1):
            self.norms.append(self.n1)
    def forward(self, x, edge_index, edge_attr):
        
        ind = 0
        for l in self.layers[:-1]:
            x = l(x, edge_index, edge_attr)
            x = self.norms[ind](x)
            ind += 1
            x = F.relu(x)
            x = F.dropout(x, 0.2)
        
        x = self.layers[-1](x, edge_index, edge_attr)
   
        return x
    
#turns edges back from format for RGCN back to original format
def transform_edge_attr(attrs):
    res = (attrs == 1).nonzero(as_tuple=False)
    t = torch.LongTensor([item[1] for item in res])
    return t

#updating X embeddings as weighted average of previous embeddings
def make_x_cluster(x, cluster_index, y_2d):
    new_size = torch.max(cluster_index)
    new_x = torch.zeros([int(new_size.item())+1, x.shape[1]])
    vals = torch.unique(cluster_index, sorted=True)
    for v in vals:
        idx = (cluster_index == v).nonzero()

        idx = idx.squeeze(1)
        if idx.shape[0] == 2:
            tmp_x =  x[idx].squeeze(1)
            tmp_y = y_2d[idx].squeeze(1)
            tot_1 = tmp_y[0][1]
            tot_2 = tmp_y[1][1]
            tmp_x[0] = tmp_x[0]*tot_1
            tmp_x[1] = tmp_x[1]*tot_2
            tmp_x = torch.sum(tmp_x, 0)
            tmp_x = tmp_x/(tot_1+tot_2)
        else:
            tmp_x = x[idx].squeeze(1)
        tmp_x = tmp_x.squeeze(0)
        new_x[int(v.item())] = tmp_x

    return new_x

def make_y_cluster(y_2d, cluster_index):
    new_size = torch.max(cluster_index)
    new_y_2d = torch.zeros([int(new_size.item())+1, 2])
    vals = torch.unique(cluster_index, sorted=True)
    for v in vals:
        idx = (cluster_index == v).nonzero()
        res = torch.sum(y_2d[idx], 0)
        res = res.squeeze(0)
        new_y_2d[int(v.item())] = res
        
    return new_y_2d

class RGCNPoolNet(torch.nn.Module):
    def __init__(self, num_RGCN_layers, min_node_score, min_edge_score):
        super(RGCNPoolNet, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.RGCN = varRGCN(num_RGCN_layers, 640)
        self.poolLayer = EdgePoolingRGCN(1, min_node_score, min_edge_score)
        #for _ in range(num_pool_layers):
            #self.layers.append(self.poolLayer)
    def forward(self, x, edge_index, edge_attr, batch, y_2d):
        pool_info = []
        print("INPUT DATA INFO")
        print("node embeddings:", x)
        print("edge index:", edge_index)
        print("edge attributes:", edge_attr)
        print("batch info", batch)
        print("====================================\n")
        print("LAYER WEIGHT INFO")
        counter = 0
        for l in self.RGCN.layers:
            print("shape of layer "+str(counter)+" weights", l.weight.shape)
            print("sum of layer "+str(counter)+" weights", torch.sum(l.weight))
            counter+=1
        print("====================================\n")
        print("OUTPUTS")

        x_1 = self.RGCN(x, edge_index, edge_attr)
        print("outputs of RGCN:", x_1)
       
        onehot_edge_attr = F.one_hot(edge_attr, 20)
        #x_bad is the computed new embeddings from edgepooling but this does not use weighted average so called "x_bad" bc we don't use it
        x_bad, outs, onehot_edge_attr, edge_index, batch, unpool = self.poolLayer(x_1, edge_index, onehot_edge_attr, batch)
        x = make_x_cluster(x, unpool.cluster, y_2d)
        y_2d = make_y_cluster(y_2d, unpool.cluster)
        
        return x, outs, unpool, y_2d, edge_index, transform_edge_attr(onehot_edge_attr), batch, x_1
    

#Main code execution
TE18_pyg = torch.load('./TE18_data_NodeEmbeddings.pt')
 
#convert dataset of pytorch geometric objects into a formt for RGCN (need a different edge format)
TE18_RGCN = []
for data in TE18_pyg:
    TE18_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

RGCN_edgePool = RGCNPoolNet(4, 0, 0)
RGCN_edgePool.load_state_dict(torch.load("./RGCNx4_edgePool_L1Sum_2xPosLabelLoss_0Minscore_percent_edge_labels_4iters_noOnehop_fixed_best_test_performance.pt"))
sys.stdout = open('RGCN_output.txt','wt')
loader = DataLoader(TE18_RGCN[:1], batch_size=1)
for d in loader:
    #y_2d is used only for the pooling part of the network
    y = d.y
    ones = torch.ones(y.shape)
    y_2d = torch.column_stack([y, ones])
    RGCN_edgePool(d.x, d.edge_index, d.edge_attr, d.batch, y_2d)

print("end of first run\n")

for d in loader:
    y = d.y
    ones = torch.ones(y.shape)
    y_2d = torch.column_stack([y, ones])
    RGCN_edgePool(d.x, d.edge_index, d.edge_attr, d.batch, y_2d)

print("end of second run\n")