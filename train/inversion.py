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
from utils import get_dataset, load_config, get_model  # Your utils



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

# def gradient_penalty(x):
#     """Compute squared gradient norm ∥∇x∥²."""
#     dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # ∂x/∂i
#     dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # ∂x/∂j

#     # Pad to match shape of x
#     dx = F.pad(dx, (0,1,0,0))  # pad right
#     dy = F.pad(dy, (0,0,0,1))  # pad bottom

#     return ((dx**2 + dy**2).mean())


laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]

def gradient_penalty(x):
    x = x.unsqueeze(1) if x.ndim == 3 else x  # ensure shape [B,1,H,W]
    weight = laplacian_kernel.to(x.device)
    lap = F.conv2d(x, weight, padding=1)
    return torch.mean(lap**2)


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

def plot_inversion_result(x0, x, true_y, y, x_pred, loss_type, index):
    """Plot inversion result for comparison with 0-centered colormap for selected plots."""
    x0_np    = x0.detach().squeeze().cpu().numpy()
    x_np     = x.squeeze().cpu().numpy()
    y_np     = y.squeeze().cpu().numpy()
    truey_np = true_y.squeeze().cpu().numpy()
    xpred_np = x_pred.detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.flatten() 
    titles = [r'Initial Guess ($a_0$)', r'Ground Truth Output ($u$)', r'Ground Truth Input ($a^\ast$)', r'Forward Prediction ($\hat{u}$)',
              r'Inversion Result ($a$)', r'$|a - a^\ast|$']
    # The first four plots will be normalized with vcenter=0.
    data = [x0_np, truey_np, x_np, y_np, xpred_np, np.abs(x_np - xpred_np)]

    for i in range(6):
        if i < 5:
            # Use TwoSlopeNorm to center the colormap at 0.
            vmin = np.min(data[i])
            vmax = np.max(data[i])
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = axes[i].imshow(data[i], cmap='jet', norm=norm)
        else:
            # For absolute error, standard normalization is used.
            vmin = np.min(data[i])
            vmax = np.max(data[i])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            im = axes[i].imshow(data[i], cmap='magma', norm=norm)
        axes[i].set_title(titles[i])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"inversion_result_{loss_type}/inversion_result_{loss_type}_{index}.png")
    plt.close(fig)


# ----------------------
# Least Squares Posterior Estimation
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate, batch_num, num_iterations=500, prior=None):
    model.eval()  # Freeze model parameters
    mse_loss = torch.nn.MSELoss()

    # x0 = torch.nn.Parameter(input_data.clone().detach().to(device))
    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []
    # Set boundaries based on the prior.
    true_min = torch.min(prior) - 0.1
    true_max = torch.max(prior) + 0.1
    print("True range:", true_min.item(), true_max.item())

    optimizer = torch.optim.Adam([x0], lr=learning_rate)
    losses, inversion_MSEs, regs, ssims = [], [], [], []

    # Global variable num_iter is assumed defined in main script.
    global num_iter
    # first_input = x0  # shape [batch, C, H, W]
    plot_single(true_data.detach().cpu().squeeze(), f'true_data.png')

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        output = model(x0)
        loss = mse_loss(output, true_data)
        # reg = torch.norm(x0)**2

        reg = gradient_penalty(x0)
        print("reg", reg.item())

        loss_total = loss + alpha * reg
        loss_total.backward()
        optimizer.step()

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
            plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}/iter={batch_num}_inversion_{iteration}.png')
            plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}/iter={batch_num}_inversion_{iteration}_output.png')
        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value)
        posterior_set.append(x0.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze()

# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 8
    num_epoch = 2101
    loss_type = "RAND"  # or "JAC"
    kernel_size = 21
    sigma = 5.0
    learning_rate = 0.001 #0.00005  # Inversion learning rate.
    num_sample = 50
    # Global regularization weight
    alpha = 1.1
    noise_std = 0.3
    
    # Load configuration and dataset.
    if loss_type == "JAC":
        config = load_config(f"output/n=128_e=8_m=FNO_s=RFS_l=JAC_lamba=0.5_20250421_221953/config.yaml")
    elif loss_type == "RAND":
        config = load_config("output/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_124311/config.yaml")
    else:
        config = load_config("output/n=128_m=FNO_l=L2_20250422_182837/config.yaml")
    data_config = load_config("output/n=32_m=FNO_l=L2_20250415_014850/config.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    
    # Here, batch_size is 1 so each iteration of the dataloader returns one sample.
    dataloader = dataset.get_dataloader(offset=128, limit=num_sample)
    
    print("Loading checkpoint")
    # Load the model using the checkpoint style.
    if loss_type == "JAC":
        ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RFS_l=JAC_lamba=0.5_20250421_221953/last.ckpt"
    elif loss_type == "RAND":
        ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_125959/last.ckpt"
    else:
        ckpt_path = "checkpoints/n=128_m=FNO_l=L2_20250422_182837/last.ckpt"
        # ckpt_path = "checkpoints/n=32_m=FNO_l=L2_20250415_123340/last.ckpt"
    model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
    print(model)

    # Initialize a list to hold loss and metric data for each sample.
    loss_data_all = []
    sample_counter = 0

    # Lists for final SSIM and L2 misfit values.
    final_ssim_list = []
    final_l2_list = []

    # Iterate over the dataloader (each batch is one sample due to batch_size=1).
    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        # Create the initial guess by applying Gaussian smoothing.
        # zero_X = apply_gaussian_smoothing(x, kernel_size, sigma) + 1e-3
        zero_X = x + torch.randn_like(x) * noise_std
        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")
        
        # Run inversion for the current sample.
        posterior_set, losses, inversion_MSEs, regs, ssims, pred = least_squares_posterior_estimation(
            model, zero_X, y, learning_rate, batch_num=sample_counter, num_iterations=num_epoch, prior=x
        )
        
        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).detach()
        plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter)
        
        # Evaluate final metrics for the current sample.
        final_ssim = ssim(
            final_x0.squeeze().cpu().numpy().astype(np.float64),
            x.squeeze().cpu().numpy().astype(np.float64),
            data_range=float(final_x0.max()-final_x0.min())
        )
        final_l2 = final_x0.squeeze().cpu().numpy() - x.squeeze().cpu().numpy()
        print(f"Sample {sample_counter} - Final SSIM: {final_ssim:.4f}, Final L2: {final_l2}")
        
        # Append final metrics to lists.
        final_ssim_list.append(final_ssim)
        final_l2_list.append(final_l2)
        
        # Record the inversion losses and metrics for every iteration for this sample.
        for itr, (loss_val, mse_val, reg_val, ssim_val) in enumerate(zip(losses, inversion_MSEs, regs, ssims)):
            loss_data_all.append({
                "sample": sample_counter,
                "iteration": itr,
                "loss": loss_val,
                "inversion_MSE": mse_val,
                "regularization": reg_val,
                "SSIM": ssim_val
            })
        
        sample_counter += 1

    # Compute and print averaged SSIM and L2 misfit over all samples.
    average_ssim = np.mean(final_ssim_list)
    average_l2 = np.mean(final_l2_list)
    print(f"\nAveraged Final SSIM over {sample_counter} sample(s): {average_ssim:.4f}")
    print(f"Averaged Final Relative L2 misfit over {sample_counter} sample(s): {average_l2:.4f}")

    # Save all loss and metric data to CSV.
    df = pd.DataFrame(loss_data_all)
    csv_file = f"loss_statistics_multiple_samples_{loss_type}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")