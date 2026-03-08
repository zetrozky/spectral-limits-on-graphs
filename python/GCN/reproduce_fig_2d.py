import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import matplotlib
import json
import networkx as nx
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_chameleon_with_noise(noise_level, seed=None):
    """
    Loads Chameleon dataset and injects noise by removing edges and adding random edges.
    Noise is applied to the unique undirected edges.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    dataset_name = 'chameleon'
    data_dir = os.path.join('new_data', dataset_name)
    
    graph_adjacency_list_file_path = os.path.join(data_dir, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(data_dir, 'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    with open(graph_node_features_and_labels_file_path) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    # Convert to PyG Data
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    
    # Ensure undirected and no self loops initially
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)

    num_nodes = len(labels)
    
    if noise_level > 0:
        # Get unique edges (upper triangle)
        mask = edge_index[0] < edge_index[1]
        unique_edges = edge_index[:, mask]
        num_unique_edges = unique_edges.size(1)
        
        num_remove = int(num_unique_edges * noise_level)
        
        # Randomly select edges to keep
        perm = torch.randperm(num_unique_edges)
        keep_idx = perm[num_remove:]
        kept_edges = unique_edges[:, keep_idx]
        
        num_add = num_remove
        
        # Add random edges
        row_rand = torch.randint(0, num_nodes, (num_add,))
        col_rand = torch.randint(0, num_nodes, (num_add,))
        
        # Remove accidental self-loops
        mask_sl = row_rand != col_rand
        row_rand = row_rand[mask_sl]
        col_rand = col_rand[mask_sl]
        
        added_edges = torch.stack([row_rand, col_rand], dim=0)
        
        # Combine
        new_edges = torch.cat([kept_edges, added_edges], dim=1)

        #ensure undirected, no duplicate edges (simple graph)
        edge_index = to_undirected(new_edges)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    # "We do not normalize features" - Appendix
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = int(y.max()) + 1
    
    return data.to(device)

def homophily(edge_index, y, method='edge'):
    row, col = edge_index
    matches = y[row] == y[col]
    return matches.float().mean().item()

class GCN_Net_1Layer(torch.nn.Module):
    """ 1 layer GCN with MSE loss (Linear output) """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

def train_and_evaluate(data, label_ratio, seed, epochs=10000):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    num_nodes = data.num_nodes
    num_classes = data.num_classes

    # Get indices for each class
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(data.y):
        class_indices[label.item()].append(idx)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)


    for c in range(num_classes):
        indices = torch.tensor(class_indices[c])
        # Shuffle indices for this class
        perm = torch.randperm(len(indices))
        indices = indices[perm]
        
        num_train_c = int(len(indices) * label_ratio)
        num_train_c = max(1, num_train_c) if label_ratio > 0 else 0
        
        train_idx_c = indices[:num_train_c]
        test_idx_c = indices[num_train_c:]
        
        train_mask[train_idx_c] = True
        test_mask[test_idx_c] = True
    
    model = GCN_Net_1Layer(data.num_features, num_classes).to(device)
    # "weight decay 10^-5" - SI Appendix
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    y_one_hot = F.one_hot(data.y, num_classes=num_classes).float().to(device)
    
    best_train_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], y_one_hot[train_mask])
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best model (minimal training loss)
    model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_mse = criterion(out[test_mask], y_one_hot[test_mask]).item()
                
    return test_mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-run', action='store_true', help='Skip training and only plot from file.')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting.')
    parser.add_argument('--results_file', type=str, default='fig_2d_results.json', help='File to save/load results.')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds to run.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    args = parser.parse_args()

    results = []
    # noise_levels = [0, 0.4, 0.8, 1.0]
    noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    label_ratios = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]

    if not args.no_run:
        seeds = range(args.seeds)
        
        print(f"Script started. Loading data...", flush=True)
        print(f"{'Noise':<10} {'LabelRatio':<12} {'TestMSE':<12} {'Std':<12} {'Homophily':<12}", flush=True)
        print("-" * 60, flush=True)

        for noise in noise_levels:
            avg_mse_per_ratio = {ratio: [] for ratio in label_ratios}
            avg_homophily = []

            for i, seed in enumerate(seeds):
                graph_seed = seed
                train_seed = seed + 100 

                data = load_chameleon_with_noise(noise, seed=graph_seed)
                
                h_val = homophily(data.edge_index, data.y)
                avg_homophily.append(h_val)
                
                if i == 0:
                     print(f"Stats for Noise {noise}: Nodes={data.num_nodes}, Edges={data.edge_index.size(1)}, "
                          f"Homophily={h_val:.4f}", flush=True)

                for ratio in label_ratios:
                    mse = train_and_evaluate(data, ratio, seed=train_seed, epochs=args.epochs)
                    avg_mse_per_ratio[ratio].append(mse)

                    print(f"  [Noise {noise}] Seed {seed}/9 | Ratio {ratio} | MSE: {mse:.4f}", flush=True)

            mean_homophily = np.mean(avg_homophily) if avg_homophily else 0.0

            for ratio in label_ratios:
                mean_mse = np.mean(avg_mse_per_ratio[ratio])
                std_mse = np.std(avg_mse_per_ratio[ratio])
                print(f"{noise:<10.2f} {ratio:<12.3f} {mean_mse:<12.4f} {std_mse:<12.4f} {mean_homophily:<12.4f}", flush=True)
                results.append({
                    'noise': noise,
                    'ratio': ratio,
                    'mse': mean_mse,
                    'std': std_mse,
                    'homophily': float(mean_homophily)
                })

        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.results_file}")

    if not args.no_plot:
        if args.no_run:
            if os.path.exists(args.results_file):
                with open(args.results_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded results from {args.results_file}")
            else:
                print(f"Results file {args.results_file} not found. Cannot plot.")
                return

        plt.style.use('default')
        
        plt.figure(figsize=(6, 5))

        plot_data = {}
        for item in results:
            noise, ratio, mse, std = item['noise'], item['ratio'], item['mse'], item.get('std', 0.0)
            if noise not in plot_data:
                plot_data[noise] = {'ratios': [], 'mses': [], 'stds': []}
            plot_data[noise]['ratios'].append(ratio)
            plot_data[noise]['mses'].append(mse)
            plot_data[noise]['stds'].append(std)

        cmap = matplotlib.colormaps.get_cmap('coolwarm')
        sorted_noise_levels = sorted(plot_data.keys())

        for noise in sorted_noise_levels:
            data = plot_data[noise]
            sorted_indices = np.argsort(data['ratios'])
            ratios = np.array(data['ratios'])[sorted_indices]
            mses = np.array(data['mses'])[sorted_indices]
            stds = np.array(data['stds'])[sorted_indices]
            
            color = cmap(noise) 
            
            label = f'{int(noise * 100)}%'
            plt.errorbar(ratios, mses, yerr=stds, label=label, color=color,
                         fmt='-o', linewidth=2, markersize=6, capsize=4, elinewidth=2, alpha=0.9)

        plt.xlabel(r'Label ratio $\tau$', fontsize=12)
        plt.ylabel('Test error', fontsize=12)
        plt.title('1 layer GCN (MSE)', fontsize=14, fontweight='bold', loc='left')
        
        plt.legend(title='Noise', bbox_to_anchor=(1, 1), loc='upper right', frameon=False)
        
        plt.xscale('log')
        plt.xticks(label_ratios, [str(r) for r in label_ratios])
        plt.minorticks_off()
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()

        output_file = 'figure_2d_reproduction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
