# GNN_graph_classification
Original paper: Wasserstein Weisfeiler-Lehman Graph Kernels (NeurIPS 2019)
This is the part of the Term Project in Data Mining. The other part is shown at [WWLsquare](https://github.com/njuxx/WWLsquare) and [WWL_RBF](https://github.com/yypurpose/WWL-RBF).
The components of data loading and metric calculation based on [the accompanying code](https://github.com/BorgwardtLab/WWL) and the GNN-based model is built by DGL. Please follow the README of that repository to install the dependencies.

## Dependencies

- `numpy`
- `scikit-learn`
- `POT`
- `cython`
- `dgl`

## Usage

```
python gnn.py --dataset your_dataset --type both --gnn edge --gp max
```

## Experiments

Here we report our result, the detailed setting is shown in our  Term Project report(to be added).

![image-20201218181004526](.\img\image-20201218181004526.png)