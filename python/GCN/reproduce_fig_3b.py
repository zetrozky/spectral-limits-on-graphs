import matplotlib.pyplot as plt
import numpy as np


def get_csbm_simulation_risk(N, F, tau_values, lambda_val, mu, r, trials=10):
    """
    Simulates Ridge Regression on CSBM with Regularization r=0.02.
    """
    # Results: rows=trials, cols=tau_values
    risks = np.zeros((trials, len(tau_values)))

    half_N = N // 2

    for t in range(trials):
        print(f"   > Trial {t + 1}/{trials}...")
        # 1. Generate Latent Labels
        y = np.ones((N, 1))
        y[half_N:] = -1

        # 2. Generate Features X (Spiked Covariance - Eq 10)
        u = np.random.randn(F, 1)
        u = u / np.linalg.norm(u)
        Z_x = np.random.randn(N, F) / np.sqrt(F)
        X = np.sqrt(mu / N) * (y @ u.T) + Z_x

        # 3. Generate Graph A (Gaussian Spiked Model - Eq 15)
        # High Lambda = Strong Homophily = Graph correlates with y
        # Symmetric (Fig 3 uses symmetric Abs)
        Z = np.random.randn(N, N)
        Xi_n = (Z + Z.T) / np.sqrt(2 * N)
        A = (lambda_val / N) * (y @ y.T) + Xi_n

        # 4. Filter Input: P(A) = A
        X_input = A @ X

        # Indices for stratified sampling
        idx_pos = np.arange(half_N)
        idx_neg = np.arange(half_N, N)
        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)

        # 5. Loop over Tau (Slice the pre-generated data)
        for i, tau in enumerate(tau_values):
            n_train_half = int((N * tau) / 2)

            if n_train_half < 1 or n_train_half >= half_N:
                risks[t, i] = np.nan
                continue

            train_idx = np.concatenate([idx_pos[:n_train_half], idx_neg[:n_train_half]])
            test_idx = np.concatenate([idx_pos[n_train_half:], idx_neg[n_train_half:]])

            X_train = X_input[train_idx, :]
            y_train = y[train_idx, :]
            X_test = X_input[test_idx, :]
            y_test = y[test_idx, :]

            reg_strength = r * tau

            # Ridge Regression with r=0.02
            # w* = (X'X + rI)^-1 X'y
            reg_matrix = X_train.T @ X_train + reg_strength * np.eye(F)

            try:
                # Use standard solve
                w_hat = np.linalg.solve(reg_matrix, X_train.T @ y_train)
            except np.linalg.LinAlgError:
                # Fallback for singularity point crashing
                w_hat = np.linalg.lstsq(reg_matrix, X_train.T @ y_train, rcond=None)[0]

            # Test Risk (MSE)
            y_pred = X_test @ w_hat
            mse = np.mean((y_pred - y_test) ** 2)
            risks[t, i] = mse

    # Average over trials
    mean_risks = np.nanmean(risks, axis=0)
    return mean_risks


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'lines.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    N = 5000
    gamma = 2.0
    F = int(N / gamma)

    r = 0.02
    mu = 1.0

    tau_values = np.linspace(0.01, 0.98, 40)

    lambda_values = [0, 1, 2, 3]
    colors = ['#2b506e', '#4db091', '#ee5d7a', '#f8cc66']

    plt.figure(figsize=(8, 6), dpi=150)
    print(f"Running CSBM Simulation (r={r})...")

    for idx, lam in enumerate(lambda_values):
        print(f" > Simulating Lambda = {lam}...")
        means = get_csbm_simulation_risk(N, F, tau_values, lam, mu, r)

        plt.plot(tau_values, means, label=r'$\lambda = {}$'.format(lam),
                 color=colors[idx], linewidth=3)

    peak_loc = 1.0 / gamma
    plt.axvline(x=peak_loc, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(peak_loc, 4.5, r'Peak $\tau=0.5$', rotation=90, ha='right', fontsize=10, color='gray')

    plt.xlabel(r'Label ratio $\tau$')
    plt.ylabel(r'Test Risk $R_{test}$')
    plt.title(r'Fig 3B Reproduction (Simulation, $r=0.02$)')
    plt.legend(title=r'Homophily $\lambda$', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(0, 5.0)
    plt.xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig('figure_3b_simulation.png')
    print("Saved figure_3b_simulation.png")


if __name__ == "__main__":
    main()
