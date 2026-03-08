import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import WebKB
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops

class ShiSAGE(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, c):
        # P(A) = A + cI
        agg = self.propagate(edge_index, x=x)
        return self.lin(agg + c * x)


def run_noise():
    print("\n Noise Test, Texas")

    dataset = WebKB(root='data/WebKB', name='Texas', transform=None)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Remove self-loops
    data.edge_index, _ = remove_self_loops(data.edge_index)

    # Noise Levels
    noise_steps = [0, 50, 100, 200, 500]
    couplings = [1.0, -1000.0]  # GCN vs MLP

    history_mean = {c: [] for c in couplings}
    history_std = {c: [] for c in couplings}

    for n_add in noise_steps:
        # Inject Noise
        if n_add > 0:
            src = torch.randint(0, data.num_nodes, (n_add,), device=device)
            dst = torch.randint(0, data.num_nodes, (n_add,), device=device)
            noise_e = torch.stack([src, dst], dim=0)
            # Make undirected for correct Laplacian eigenvalues
            noise_e = torch.cat([noise_e, noise_e.flip(0)], dim=1)
            current_edges = torch.cat([data.edge_index, noise_e], dim=1)
        else:
            current_edges = data.edge_index

        print(f"Noise +{n_add} edges:")

        for c in couplings:
            split_accs = []

            # Loop over all 10 canonical splits
            for i in range(10):
                mask_train = data.train_mask[:, i]
                mask_test = data.test_mask[:, i]

                model = ShiSAGE(dataset.num_features, dataset.num_classes).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                # Train
                model.train()
                for _ in range(200):
                    opt.zero_grad()
                    out = model(data.x, current_edges, c=c)
                    loss = F.cross_entropy(out[mask_train], data.y[mask_train])
                    loss.backward()
                    opt.step()

                # Eval
                model.eval()
                out = model(data.x, current_edges, c=c)
                pred = out.argmax(dim=1)
                acc = (pred[mask_test] == data.y[mask_test]).float().mean().item()
                split_accs.append(acc)

            # Aggregate stats
            mu = np.mean(split_accs)
            sigma = np.std(split_accs)
            history_mean[c].append(mu)
            history_std[c].append(sigma)
            print(f"  c={c:<7} | Acc: {mu:.4f} (+/- {sigma:.4f})")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.errorbar(noise_steps, history_mean[1.0], yerr=history_std[1.0], fmt='o-', label='GCN (c=1.0)', capsize=5)
    plt.errorbar(noise_steps, history_mean[-1000.0], yerr=history_std[-1000.0], fmt='x--', color='red',
                 label='MLP Limit (c=-inf)', capsize=5)
    plt.xlabel("Added Noise Edges")
    plt.ylabel("Test Accuracy")
    plt.title("Noise on Texas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("audit_2_noise.png")
    print("Saved audit_2_noise.png")


if __name__ == "__main__":
    run_noise()
