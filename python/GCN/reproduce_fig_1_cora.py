import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_cora():
    dataset = Planetoid(root='data/Planetoid', name='Cora') 
    data = dataset[0].to(device)
    return dataset, data

# Model Classes GCN_Net0 to GCN_Net3 remain the same
class GCN_Net0(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_classes)
    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)

class GCN_Net1(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GCN_Net2(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# Net3 is structurally identical to Net2, just trained with CE loss
class GCN_Net3(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

def get_stratified_masks(data, label_ratio, num_classes):
    """
    FIX: Stratified sampling to ensure every label is represented, 
    matching the paper's logic to avoid zero-sample pathology.
    """
    N = data.num_nodes
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    # Get indices for each class
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(data.y):
        class_indices[label.item()].append(idx)
    
    for c in range(num_classes):
        indices = torch.tensor(class_indices[c])
        perm = torch.randperm(len(indices))
        indices = indices[perm]
        
        # Calculate specific count for this class based on global ratio
        # Note: For very small ratios, we enforce at least 1 sample if ratio > 0
        num_train_c = int(len(indices) * label_ratio)
        if label_ratio > 0 and num_train_c == 0:
            num_train_c = 1 # Ensure at least one sample per class
            
        train_idx_c = indices[:num_train_c]
        test_idx_c = indices[num_train_c:]
        
        train_mask[train_idx_c] = True
        test_mask[test_idx_c] = True
        
    return train_mask.to(device), test_mask.to(device)

def train_and_evaluate(data, label_ratio, seed, net_type, num_classes, epochs=10000):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Use Stratified Sampling
    train_mask, test_mask = get_stratified_masks(data, label_ratio, num_classes)
    
    # Model Selection
    if net_type == 0:
        model = GCN_Net0(data.num_features, num_classes).to(device)
        criterion = torch.nn.MSELoss()
        is_mse = True
    elif net_type == 1:
        model = GCN_Net1(data.num_features, num_classes).to(device)
        criterion = torch.nn.MSELoss()
        is_mse = True
    elif net_type == 2:
        model = GCN_Net2(data.num_features, num_classes).to(device)
        criterion = torch.nn.MSELoss()
        is_mse = True
    elif net_type == 3:
        model = GCN_Net3(data.num_features, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        is_mse = False
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    y_target = data.y
    if is_mse:
        y_target = F.one_hot(data.y, num_classes=num_classes).float().to(device)
        
    best_train_loss = float('inf')
    test_loss_at_best_train = float('inf')
    test_acc_at_best_train = 0.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        if is_mse:
            loss = criterion(out[train_mask], y_target[train_mask])
        else:
            loss = criterion(out[train_mask], data.y[train_mask]) # CE expects indices
            
        loss.backward()
        optimizer.step()
        
        # Check training loss
        train_loss = loss.item()
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            
            # Evaluate only when we find a better model
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                if is_mse:
                    test_loss = criterion(out[test_mask], y_target[test_mask]).item()
                else:
                    test_loss = criterion(out[test_mask], data.y[test_mask]).item()
                
                pred = out.argmax(dim=1)
                correct = (pred[test_mask] == data.y[test_mask]).sum()
                # Safe division check
                test_total = int(test_mask.sum())
                test_acc = int(correct) / test_total if test_total > 0 else 0.0
                
                test_loss_at_best_train = test_loss
                test_acc_at_best_train = test_acc
                
    return test_loss_at_best_train, test_acc_at_best_train

def main():
    dataset, data = load_cora()
    num_classes = dataset.num_classes

    label_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    
    seeds = range(10) 
    
    configs = [
        {'net': 0, 'title': '1 layer, linear, MSE'},
        {'net': 1, 'title': '2 layers, ReLU, MSE'},
        {'net': 2, 'title': '2 layers, ReLU, MSE, dropout'},
        {'net': 3, 'title': '2 layers, ReLU, CE, dropout'}
    ]
    
    results = {}
    
    for config in configs:
        net = config['net']
        print(f"Running {config['title']}...")
        
        avg_losses = []
        std_losses = []
        avg_accs = []
        std_accs = []
        
        for ratio in tqdm(label_ratios):
            losses = []
            accs = []
            for seed in seeds:
                l, a = train_and_evaluate(data, ratio, seed, net, num_classes, epochs=10000)
                losses.append(l)
                accs.append(a)
            
            avg_losses.append(np.mean(losses))
            std_losses.append(np.std(losses))
            avg_accs.append(np.mean(accs))
            std_accs.append(np.std(accs))
            
        results[net] = {
            'avg_losses': avg_losses,
            'std_losses': std_losses,
            'avg_accs': avg_accs,
            'std_accs': std_accs
        }
        
    # Plotting Logic (Same as before, ensures structure matches)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, config in enumerate(configs):
        net = config['net']
        ax1 = axes[i]
        res = results[net]
        ratios = label_ratios
        
        # Plot Test Error (Red)
        color = 'tab:red'
        ax1.set_xlabel(r'Label ratio $\tau$')
        if i == 0:
            ax1.set_ylabel('Test error', color=color)
        ax1.plot(ratios, res['avg_losses'], color=color, marker='o', label='Test error')
        ax1.fill_between(ratios, 
                         np.array(res['avg_losses']) - np.array(res['std_losses']),
                         np.array(res['avg_losses']) + np.array(res['std_losses']),
                         color=color, alpha=0.2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')
        ax1.set_xticks(ratios)
        ax1.set_xticklabels([str(r) for r in ratios], rotation=45)
        
        # Plot Accuracy (Black)
        ax2 = ax1.twinx()
        color = 'black'
        if i == 3:
            ax2.set_ylabel('Accuracy', color=color)
        else:
            ax2.set_yticks([]) 
            
        ax2.plot(ratios, res['avg_accs'], color=color, marker='o', label='Accuracy')
        ax2.fill_between(ratios, 
                         np.array(res['avg_accs']) - np.array(res['std_accs']),
                         np.array(res['avg_accs']) + np.array(res['std_accs']),
                         color=color, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title(config['title'])
        
    plt.tight_layout()
    plt.savefig('figure_1_cora_reproduction_fixed.png')
    print("Plot saved to figure_1_cora_reproduction_fixed.png")

if __name__ == "__main__":
    main()