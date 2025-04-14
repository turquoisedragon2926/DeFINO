import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from modulus.models.fno import FNO
import h5py
from skimage.metrics import structural_similarity as ssim
from matplotlib import colors
from scipy.interpolate import interp1d
import sys
import torch.nn.functional as F

from models.ns import NSModel  # Your model
from utils import get_dataset, load_config  # Your utils

# Global regularization weight
alpha = 0.00001

# ----------------------
# Gaussian Smoothing Functions
# ----------------------
def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] @ gauss[None, :]
    return kernel

def apply_gaussian_smoothing(batch_matrix: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing to a batch of input matrices using a Gaussian kernel."""
    kernel = gaussian_kernel(kernel_size, sigma).to(batch_matrix.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x k x k
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    original_min = batch_matrix.amin(dim=(-2, -1), keepdim=True)
    original_max = batch_matrix.amax(dim=(-2, -1), keepdim=True)
    original_range = original_max - original_min

    smoothed_batch = F.conv2d(batch_matrix, kernel, padding=kernel_size // 2, groups=1)
    smoothed_min = smoothed_batch.amin(dim=(-2, -1), keepdim=True)
    smoothed_max = smoothed_batch.amax(dim=(-2, -1), keepdim=True)
    smoothed_range = smoothed_max - smoothed_min

    rescaled_batch = (smoothed_batch - smoothed_min) / (smoothed_range + 1e-8) * original_range + original_min
    return rescaled_batch

# ----------------------
# Plotting Functions
# ----------------------
def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_inversion_result(x0, x, y, x_pred):
    """Plot inversion result for comparison."""
    x0_np = x0.detach().squeeze().cpu().numpy()
    x_np = x.squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()
    xpred_np = x_pred.detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(5, 1, figsize=(10, 20))
    titles = ['Initial Guess (x0)', 'Target Output (y)', 'Ground Truth Input (x)', 
              'Inverted Input (x_pred)', 'Error |x - x_pred|']
    data = [x0_np, y_np, x_np, xpred_np, np.abs(x_np - xpred_np)]

    for i in range(5):
        im = axes[i].imshow(data[i], cmap='jet' if i < 4 else 'magma')
        axes[i].set_title(titles[i])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("inversion_result.png")
    plt.close(fig)

# ----------------------
# Least Squares Posterior Estimation (MLE) Function
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate, batch_num, num_iterations=500, prior=None):
    model.eval()  # Freeze model parameters
    mse_loss = torch.nn.MSELoss()

    x0 = torch.nn.Parameter(input_data.clone().detach().to(device))
    
    posterior_set = []
    # Set boundaries based on the prior.
    true_min = torch.min(prior) - 0.1
    true_max = torch.max(prior) + 0.1
    print("True range:", true_min.item(), true_max.item())

    optimizer = torch.optim.Adam([x0], lr=learning_rate)
    losses, inversion_MSEs, regs, ssims = [], [], [], []

    # Global variable num_iter is assumed defined in main script.
    global num_iter

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        # Propagate input through the model repeatedly (simulate multi-step process)
        first_input = x0  # shape [batch, C, H, W]
        for ts in range(num_iter):
            output = model(first_input)
            first_input = output
        # output retains shape [batch, C, H, W]
        loss = mse_loss(output, true_data)
        reg = torch.norm(x0)**2
        loss_total = loss + alpha * reg

        loss_total.backward()
        optimizer.step()

        # Optionally clamp x0 to prior range
        with torch.no_grad():
            x0.data.clamp_(min=true_min, max=true_max)

        losses.append(loss_total.item())
        inversion_MSE = F.mse_loss(x0, prior)
        inversion_MSEs.append(inversion_MSE.item())
        regs.append(reg.item())
        input_numpy = x0.detach().cpu().squeeze().numpy()
        prior_numpy = prior.detach().cpu().squeeze().numpy()
        ssim_value = ssim(input_numpy.astype(np.float64), prior_numpy.astype(np.float64),
                          data_range=float(input_numpy.max()-input_numpy.min()))
        ssims.append(ssim_value)

        if batch_num < 2 and iteration % 50 == 0:
            plot_single(x0.detach().cpu().squeeze(), f'iter={batch_num}_inversion_{iteration}.png')
        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}")
        posterior_set.append(x0.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims

# ----------------------
# Main Script
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 10
    num_epoch = 601
    discretization = 128
    kernel_size = 27
    sigma = 5.0
    learning_rate = 0.002  # Inversion learning rate.
    
    # Load configuration and dataset.
    config = load_config("output/n=64_e=1_m=FNO_s=RFS_l=JAC_lamba=0.5_20250409_124642/config.yaml")
    dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
    
    # Get a batch: use offset equals the number of training samples, limit 1, shuffle=False.
    dataloader = dataset.get_dataloader(offset=config.training_settings.num_train, limit=1, shuffle=False)
    batch = next(iter(dataloader))
    
    # Define x ("prior"/ground truth) and y (target output).
    x = batch['x'].to(device).float()
    y = batch['y'].to(device).float()
    
    # Apply Gaussian smoothing to x for the initial guess.
    zero_X = apply_gaussian_smoothing(x, kernel_size, sigma)
    plot_single(zero_X.detach().cpu().squeeze(), "zero_X.png", "jet")
    
    # Global variable: number of inner time-steps in inversion.
    num_datapoint = 1000
    num_init = 100
    num_iter = int(num_datapoint / num_init)
    
    # Load the model using ckpt style.
    ckpt_path = "checkpoints/n=64_e=1_m=FNO_s=RFS_l=JAC_lamba=0.5_20250409_124642/n=64_e=1_m=FNO_s=RFS_l=JAC_lamba=0.5_epoch=047_val_rel_l2_loss=0.4138.ckpt"
    model = NSModel(**config.model_settings, ckpt_path=ckpt_path).eval().to(device)
    
    # Run inversion using the single model.
    # Here, we set the prior for inversion to be x.
    posterior_set, losses, inversion_MSEs, regs, ssims = least_squares_posterior_estimation(
        model, zero_X, x, learning_rate, batch_num=0, num_iterations=num_epoch, prior=x
    )
    
    # Plot and print final evaluation metrics.
    final_x0 = torch.tensor(posterior_set[-1]).detach()
    final_ssim = ssim(
        final_x0.squeeze().cpu().numpy().astype(np.float64),
        x.squeeze().cpu().numpy().astype(np.float64),
        data_range=float(final_x0.max()-final_x0.min())
    )
    final_l2 = np.linalg.norm(final_x0.squeeze().cpu().numpy() - x.squeeze().cpu().numpy()) / np.linalg.norm(x.squeeze().cpu().numpy())
    print(f"Final SSIM: {final_ssim:.4f}")
    print(f"Final Rel. L2: {final_l2:.4f}")
    
    # Plot final inversion result.
    plot_inversion_result(zero_X, x, y, final_x0)
    
    # Optionally, save loss curves to CSV.
    loss_data = {
        "Loss": losses,
        "Inversion MSE": inversion_MSEs,
        "Regularization": regs,
        "SSIM": ssims
    }
    df = pd.DataFrame(loss_data)
    csv_file = "loss_statistics.csv"
    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")
