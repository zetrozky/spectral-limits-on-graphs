import io

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, remove_self_loops


# Data loader
def load_platonov_squirrel(device):
    print("\n[Protocol] Fetching Filtered Squirrel (Platonov et al., 2023)...")
    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/squirrel_filtered.npz"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error downloading: {e}")
        exit(1)

    with np.load(io.BytesIO(response.content)) as f:
        x = torch.from_numpy(f['node_features']).float()
        y = torch.from_numpy(f['node_labels']).long()
        edge_index = torch.from_numpy(f['edges']).t().contiguous().long()

        # Transpose (.t()) masks from (Splits, Nodes) -> (Nodes, Splits)
        train_masks = torch.from_numpy(f['train_masks']).t().contiguous().bool()
        val_masks = torch.from_numpy(f['val_masks']).t().contiguous().bool()
        test_masks = torch.from_numpy(f['test_masks']).t().contiguous().bool()

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

    print(f"[Status] Loaded Filtered Squirrel: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data.to(device)



class SurgerySAGE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, data):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

        print("Computing Spectrum...")
        # 1. Compute Normalized Laplacian
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        # Symmetrize
        adj = torch.maximum(adj, adj.t())

        deg = adj.sum(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # L = I - D^-1/2 A D^-1/2
        self.L = torch.eye(data.num_nodes, device=adj.device) - deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        # 2. Eigendecomposition
        self.evals, self.evecs = torch.linalg.eigh(self.L)
        print("Spectrum Computed.")

    def forward(self, x, mode, epsilon=0.75):
        # 1. Fourier Transform
        x_spec = self.evecs.t() @ x

        # 2. Apply Filter Mask
        mask = torch.ones_like(self.evals)

        if mode == 'mlp':
            # Identity pass-through (ignores graph structure)
            pass

        elif mode == 'high_pass':  # Cut everything below lambda=1.0
            mask[self.evals < 1.0] = 0.0

        elif mode == 'band_stop':
            # Cut out the center
            dist_from_one = torch.abs(self.evals - 1.0)

            # epsilon removes range [0.25, 1.75]
            mask[dist_from_one < epsilon] = 0.0

            # 3. Inverse Fourier
        x_filt = self.evecs @ (mask.view(-1, 1) * x_spec)
        return self.lin(x_filt)

def run_surgery():
    print("\nRunning band-stop filter on Platonov Squirrel")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load
    data = load_platonov_squirrel(device)

    # 2. Sanitize 
    data.edge_index, _ = remove_self_loops(data.edge_index)

    model = SurgerySAGE(data.num_features, data.y.max().item() + 1, data).to(device)

    modes = ['mlp', 'high_pass', 'band_stop']
    results = {m: [] for m in modes}

    num_splits = 10
    print(f"\nAveraging over {num_splits} splits.")

    for split_idx in range(num_splits):
        print(f"\n--- Split {split_idx} ---")

        for mode in modes:
            model.lin.reset_parameters()
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            # Tracking metrics
            best_val_acc = 0
            final_test_acc = 0

            for epoch in range(200):
                model.train()
                opt.zero_grad()
                out = model(data.x, mode=mode)

                # Training Loss
                loss = F.cross_entropy(
                    out[data.train_mask[:, split_idx]],
                    data.y[data.train_mask[:, split_idx]]
                )
                loss.backward()
                opt.step()

                # Validation & Test
                model.eval()
                with torch.no_grad():
                    pred = out.argmax(1)

                    # 1. Validation Accuracy
                    val_acc = (pred[data.val_mask[:, split_idx]] == data.y[
                        data.val_mask[:, split_idx]]).float().mean().item()

                    # 2. Test Accuracy
                    test_acc = (pred[data.test_mask[:, split_idx]] == data.y[
                        data.test_mask[:, split_idx]]).float().mean().item()

                    # 3. Model Selection
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc

            results[mode].append(final_test_acc)
            print(f"  Mode {mode.ljust(10)} | Val: {best_val_acc:.4f} | Test: {final_test_acc:.4f}")

    print("\n--- Audit Results (Mean ± Std) ---")
    for mode in modes:
        avg = torch.tensor(results[mode]).mean().item()
        std = torch.tensor(results[mode]).std().item()
        print(f"{mode.ljust(10)}: {avg:.4f} ± {std:.4f}")

    band_stop_avg = torch.tensor(results['band_stop']).mean()
    mlp_avg = torch.tensor(results['mlp']).mean()

    print("\n--- Interpretation ---")
    if band_stop_avg > mlp_avg:
        print(f"Works! Band-Stop ({band_stop_avg:.4f}) beats MLP ({mlp_avg:.4f}).")
    else:
        print(f"Doesn't work! Band-Stop ({band_stop_avg:.4f}) failed to beat MLP ({mlp_avg:.4f}).")


if __name__ == "__main__":
    run_surgery()
