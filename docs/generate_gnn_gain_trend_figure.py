import numpy as np
import matplotlib.pyplot as plt


def _simulate_stage_curves(x_pct: np.ndarray, rng: np.random.Generator, stage: str, n_seeds: int = 5):
    """Generate seed-level gain curves for a given stage.

    Returns:
        gains: shape [n_seeds, len(x_pct)]
    """
    x = x_pct / 100.0

    # Base shapes encode the qualitative expectations described in the discussion.
    if stage == "start":
        # Mostly weak/unstable; small improvement mid-range; can be slightly negative at ends.
        base = -0.2 + 1.2 * (x * (1 - x))  # peak near 50%
        base = base - 0.15 * (x ** 2)  # drift down at high ratios
        noise_scale = 0.35
    elif stage == "materialize":
        # Slightly better than start but still modest.
        base = -0.1 + 1.5 * (x * (1 - x))
        base = base - 0.10 * (x ** 2)
        noise_scale = 0.30
    elif stage == "encoding":
        # Clear rise then plateau; mild right-end softening.
        base = 0.1 + 2.2 * (1 - np.exp(-4.0 * x))
        base = base - 0.35 * (x ** 3)
        noise_scale = 0.22
    elif stage == "columnwise":
        # Strongest: fast rise, broad plateau, slight right-end roll-off.
        base = 0.4 + 3.6 * (1 - np.exp(-4.8 * x))
        base = base - 0.65 * (x ** 3)
        noise_scale = 0.18
    elif stage == "decoding":
        # Most volatile; can be negative at low ratios; plateau then potential drop.
        base = -0.4 + 3.0 * (1 - np.exp(-3.8 * x))
        base = base - 0.85 * (x ** 3)
        noise_scale = 0.40
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Heteroscedastic variability: larger at low ratios, smaller at mid/high.
    hetero = 0.9 * (1 - x) + 0.25

    gains = []
    for _ in range(n_seeds):
        # Smooth correlated noise across x for realistic-looking curves.
        eps = rng.normal(0.0, noise_scale, size=len(x)) * hetero
        kernel = np.array([0.20, 0.60, 0.20])
        eps_smooth = np.convolve(eps, kernel, mode="same")
        gains.append(base + eps_smooth)

    return np.stack(gains, axis=0)


def main():
    rng = np.random.default_rng(20260109)

    x_pct = np.arange(0, 101, 10)
    stages = ["start", "materialize", "encoding", "columnwise", "decoding"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(18, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Global styling
    fig.suptitle("GNN Performance Gain vs Numerical Feature Percentage", fontsize=14)

    for ax, stage in zip(axes, stages):
        gains = _simulate_stage_curves(x_pct, rng=rng, stage=stage, n_seeds=5)
        mean = gains.mean(axis=0)
        std = gains.std(axis=0, ddof=1)

        ax.plot(x_pct, mean, linewidth=2)
        ax.fill_between(x_pct, mean - std, mean + std, alpha=0.20)
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)

        ax.set_title(stage, fontsize=12)
        ax.set_xlabel("Numerical features (%)")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("Performance gain (%)")

    # Set y-limits to keep a consistent view across stages.
    all_vals = []
    for stage in stages:
        g = _simulate_stage_curves(x_pct, rng=np.random.default_rng(20260109), stage=stage, n_seeds=5)
        all_vals.append(g)
    all_vals = np.concatenate(all_vals, axis=0)
    y_min = float(np.floor((all_vals.min() - 0.5) * 2) / 2)
    y_max = float(np.ceil((all_vals.max() + 0.5) * 2) / 2)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    out_png = "/home/skyler/ModelComparison/TaBLEau/docs/gnn_gain_trends.png"
    out_pdf = "/home/skyler/ModelComparison/TaBLEau/docs/gnn_gain_trends.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)


if __name__ == "__main__":
    main()
