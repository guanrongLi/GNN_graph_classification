from dgl.utils import expand_as_pair
from utilities import load_gnn_graphs,load_continuous_graphs
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import *
from torch.utils.data import DataLoader
from dgl.base import DGLError
from argparse import ArgumentParser
import os


class MyEdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(MyEdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def message(self, edges):
        """The message computation function.
        """
        # theta_x = self.theta(edges.data['node_labels'] * (edges.dst['x'] - edges.src['x']))
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            if not self.batch_norm:
                g.update_all(self.message, fn.max('e', 'x'))
            else:
                g.apply_edges(self.message)
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
            return g.dstdata['x']


def collate_graph(data):
    graphs, labels = map(list, zip(*data))
    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    return bg, labels


class GNN_cls(torch.nn.Module):
    def __init__(self, num_features, num_classes, gnn, gp):
        super(GNN_cls, self).__init__()
        if gnn == 'edge':
            conv = MyEdgeConv
        else:
            conv = GraphConv
        if gp == 'max':
            self.pooling = MaxPooling()
            self.fc1 = torch.nn.Linear(64, 64)
        else:
            self.pooling = Set2Set(64, 4, 2)
            self.fc1 = torch.nn.Linear(64 * 2, 64)
        self.conv1 = conv(num_features, 32, allow_zero_in_degree=True, batch_norm=True)
        self.conv2 = conv(32, 64, allow_zero_in_degree=True)
        self.conv3 = conv(64, 64, allow_zero_in_degree=True)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except Exception as e:
                print(e)
                continue

    def forward(self, graph, atom_feats):
        x = F.elu(self.conv1(graph, atom_feats))
        x = F.elu(self.conv2(graph, x))
        x = F.elu(self.conv3(graph, x))
        x = self.pooling(graph, x)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='DD', help='Provide the dataset name')
    parser.add_argument('--crossvalidation', default=False, action='store_true',
                        help='Enable a 10-fold crossvalidation')
    parser.add_argument('--type', type=str, default='continuous')
    parser.add_argument('--gnn', type=str, default='edge',
                        help='Provide the gnn layer types, edgeconv or graphconv')
    parser.add_argument('--gp', type=str, default='max', help='Provide the graph pooling method, max or set')

    args = parser.parse_args()
    dataset = args.dataset
    typ = args.type
    gnn = args.gnn
    gp = args.gp

    if typ != 'discrete' and typ != 'continuous' and typ != 'both' and gnn != 'edge' and gnn != 'graph' and gp != 'max' and gp != 'set':
        print('Type error!')
        exit(-1)
    print(f'Generating results for {dataset}...')
    # ---------------------------------
    # Setup
    # ---------------------------------
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)

    for path in [output_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # ---------------------------------
    # Embeddings
    # ---------------------------------
    node_labels, node_features, adj_mat, n_nodes, edge_features, y, graphs = load_continuous_graphs(dataset)
    n_classes = len(set(y))
    n_nodes_labels = 0
    for g in node_labels:
        n_nodes_labels = int(max(n_nodes_labels, max(g)))
    labels_emb = nn.Embedding(n_nodes_labels + 1, 64)
    data, graphs, n_node_embed = load_gnn_graphs(dataset, labels_emb, typ)

    # random the dataset
    perm = torch.randperm(len(data), dtype=torch.long)
    all_data = data.index_select(perm)
    all_data = list(map(list, all_data))
    if dataset in ["ENZYMES", "PROTEINS", "NCI1"]:
        for i in range(len(all_data)):
            # only used in ENZYMES and PROTEINS
            all_data[i][0] = dgl.add_self_loop(all_data[i][0])
    dataset = all_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN_cls(n_node_embed, n_classes, gnn, gp).to(device)

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0

        for data in loader:
            graph, y = data
            # y = F.one_hot(y.long(), num_classes=n_classes)
            graph = graph.to(device)
            atom_feats = graph.ndata.pop('node_embed').float().to(device)
            y = y.long().squeeze().to(device)

            optimizer.zero_grad()
            out_probility = model(graph, atom_feats)
            loss = F.nll_loss(out_probility, y, reduction='sum')
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        return loss_all / len(loader.dataset)

    def val(loader):
        model.eval()
        correct = 0
        loss_all = 0

        for data in loader:
            graph, y = data
            graph = graph.to(device)
            atom_feats = graph.ndata.pop('node_embed').float().to(device)
            y = y.long().squeeze().to(device)
            pred = model(graph, atom_feats).max(1)[1]
            correct += pred.eq(y).sum().item()
            loss_all += F.nll_loss(model(graph, atom_feats), y, reduction='sum').item()
        return correct / len(loader.dataset), loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        correct = 0

        for data in loader:
            graph, y = data
            graph = graph.to(device)
            atom_feats = graph.ndata.pop('node_embed').float().to(device)
            y = y.squeeze().to(device)
            pred = model(graph, atom_feats).max(1)[1]
            correct += pred.eq(y).sum().item()
        return correct / len(loader.dataset)

    # ---------------------------------
    acc = []
    n = len(data) // 10
    for i in range(10):
        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        n = len(data) // 10
        test_dataset = dataset[i * n:(i + 1) * n]
        train_dataset = dataset[0: i * n] + dataset[n * (i + 1):]

        n = len(train_dataset) // 10
        val_dataset = train_dataset[i * n:(i + 1) * n]
        val_data = DataLoader(val_dataset, batch_size=128, collate_fn=collate_graph)
        test_data = DataLoader(test_dataset, batch_size=128, collate_fn=collate_graph)
        train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_graph)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)
        best_val_loss, test_acc = 100, 0
        patience, now_step = 10, 0
        for epoch in range(1, 101):
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(epoch, train_data, optimizer)
            val_acc, val_loss = val(val_data)
            scheduler.step(val_loss)
            if best_val_loss >= val_loss:
                test_acc = test(test_data)
                best_val_loss = val_loss
                now_step = 0
            else:
                now_step += 1
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                  'Val Loss: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(
                epoch, lr, train_loss, val_loss, val_acc, test_acc))
            if now_step > patience:
                print('Early stop at Epoch: {:03d}'.format(epoch))
                break
        acc.append(test_acc)
    acc = torch.tensor(acc)
    print('---------------- Final Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))


if __name__ == '__main__':
    main()
