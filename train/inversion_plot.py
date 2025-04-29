import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_comparison(csv_jac, csv_mse, csv_rand, metrics, color_map, output_filename, metric_name):
    """
    Loads three CSV files (JAC, MSE, RAND), computes mean and std for each iteration,
    and plots each metric on a separate subplot for comparison.
    """
    # Load data
    df_jac = pd.read_csv(csv_jac)
    df_mse = pd.read_csv(csv_mse)
    df_rand = pd.read_csv(csv_rand)
    
    # Layout
    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5*nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Aggregate by iteration
        grouped_jac = df_jac.groupby("iteration")[metric].agg(["mean", "std"]).reset_index()
        grouped_mse = df_mse.groupby("iteration")[metric].agg(["mean", "std"]).reset_index()
        grouped_rand = df_rand.groupby("iteration")[metric].agg(["mean", "std"]).reset_index()
        
        # Unpack colors
        color_jac, color_mse, color_rand = color_map.get(metric, ("blue", "orange", "gray"))
        
        # Plot JAC
        ax.plot(grouped_jac["iteration"], grouped_jac["mean"],
                label="Jvp:FIM", color=color_jac, marker='o', markevery=50, linewidth=2)
        ax.fill_between(grouped_jac["iteration"], grouped_jac["mean"] - 2 * grouped_jac["std"],
                        grouped_jac["mean"] + 2 * grouped_jac["std"], color=color_jac, alpha=0.2)
        
        # Plot RAND
        ax.plot(grouped_rand["iteration"], grouped_rand["mean"],
                label="Jvp:RAND", color=color_rand, marker='^', markevery=50, linewidth=2)
        ax.fill_between(grouped_rand["iteration"], grouped_rand["mean"] - 2 * grouped_rand["std"],
                        grouped_rand["mean"] + 2 * grouped_rand["std"], color=color_rand, alpha=0.2)
        
        # Plot MSE
        ax.plot(grouped_mse["iteration"], grouped_mse["mean"],
                label="MSE", color=color_mse, marker='s', markevery=50, linewidth=2)
        ax.fill_between(grouped_mse["iteration"], grouped_mse["mean"] - 2 * grouped_mse["std"],
                        grouped_mse["mean"] + 2 * grouped_mse["std"], color=color_mse, alpha=0.2)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(metric_name[idx], fontsize=12)
        ax.set_title(f"{metric_name[idx]}", fontsize=14)
        ax.legend(fontsize=12)
    
    # Clean up empty plots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.show()

# Files
csv_jac = "loss_statistics_multiple_samples_JAC.csv"
csv_mse = "loss_statistics_multiple_samples_MSE.csv"
csv_rand = "loss_statistics_multiple_samples_RAND.csv"

# Metrics
metrics = ["loss", "inversion_MSE", "regularization", "SSIM"]
# Different metrics
metric_name = ["Forward Error", "Inversion Error", "Regularization", "SSIM"]

# Global setting of tick size
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Color map: (JAC, MSE, RAND)
color_map = {
    "loss": ("#1f77b4", "#aec7e8", "#7f7f7f"),
    "inversion_MSE": ("#ff7f0e", "#ffbb78", "#c7c7c7"),
    "regularization": ("#2ca02c", "#98df8a", "#bcbd22"),
    "SSIM": ("#d62728", "#ff9896", "#8c564b")
}

# Plot
plot_metrics_comparison(csv_jac, csv_mse, csv_rand, metrics, color_map, "metrics_comparison.png", metric_name)
