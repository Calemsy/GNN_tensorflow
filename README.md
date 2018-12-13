# Deep Graph Convolutional Neural Network implement by tensorflow

## About

GNN is a novel and powerful deep neural network for graph classification, It usually consists of *(1)*`graph convolution layer` which extract local substructure features for individual links and *(2)* a `SortPooling layer` which aggregates node-level features into a graph-level feature vector. It's directly accepts graph data as input without the need of first transforming graphs into tensors, make end-to-end gradient-based training possible. And it enables learning from global topology by sorting the vertex features instead of summing them up, which is supportd by `SortPooling layer`.

This is the implementation based on **Tensorflow**.

For more information, please refer to:

> M. Zhang, Z. Cui, M. Neumann, and Y. Chen, An End-to-End Deep Learning Architecture for Graph Classification, Proc. AAAI Conference on Artificial Intelligence (AAAI-18).

and the origal PyTorch implementation of DGCNN is [here](https://github.com/muhanzhang/pytorch_DGCNN)

## Result

| **Dataset**  | Mutag  | NCI1 | PROTEINS |
| :------------ |:---------------:| :-----:|:-----:|
| Nodes(max)     |   28  | 111 | 620 |  
| Nodes(avg.)     |    17.93    |  29.87  | 39.06 |
| Nodes(min)     |      10   |  3 | 4 |
| Graphs    |      188   |  4110 | 1113 |
| **GNN** | **0.8684**(0.058844)| **0.7073**(0.018595)| **0.7509**(0.027505)|

To be continued...
