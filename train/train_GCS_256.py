import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import datetime
import time
import numpy as np
import argparse
import json
import logging
import os
import csv
import h5py
import pandas as pd
import tqdm
import math
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sGCS
from functorch import vjp, vmap
from torch.utils.data import Subset
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interpn

from torch.utils.data import Dataset, DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected

class Timer:
    def __init__(self):
        self.elapsed_times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_times.append(self.elapsed_time)
        return False


### Dataset ###
class GCSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

### Auxiliary Function ###
def save_dataset_to_csv(dataset, prefix):
    k_data = []
    
    for k in dataset:
        k_data.append(k)
    
    k_df = pd.DataFrame(k_data)
    k_df.to_csv(f'{prefix}', index=False)
    print(f"Saved {prefix} dataset to CSV files")
    return

def load_dataset_from_csv(prefix, nx, ny):
    df = pd.read_csv(f'{prefix}')

    print(f'df Length: {len(df)}')
    print(f'df Shape: {df.shape}')
    
    data = [torch.tensor(row.values).reshape(nx, ny) for _, row in df.iterrows()]
    
    return data


### Compute Metric ###
def plot_results(true1, pred1, path):
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 3, 1)
    plt.imshow(true1.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True Saturation')

    plt.subplot(1, 3, 2)
    plt.imshow(pred1.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted Saturation')

    # Set colorbar to be centered at 0 for error map
    plt.subplot(1, 3, 3)
    error1 = true1.cpu().numpy() - pred1.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error1.min()), abs(error1.max()))
    plt.imshow(np.abs(error1), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Create a centered normalization around 0
    norm = colors.CenteredNorm()

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(ax, fraction=0.045, pad=0.06, norm=norm)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single_abs(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap)
    plt.colorbar(ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        norm = colors.CenteredNorm()
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, norm=norm)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        im = ax.imshow(true1, cmap=cmap)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list=None):
    # Create loss plot
    print("Create loss plot")
    if epoch < 510:
        mse_diff = np.asarray(mse_diff)
        jac_diff_list = np.asarray(jac_diff_list)
        test_diff = np.asarray(test_diff)
    else: 
        start_epoch = 30
        mse_diff = mse_diff[start_epoch:]
        test_diff = test_diff[start_epoch:]
        jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant

    path = f"../plot/Loss/Checkpoint/FNO_GCS_vec:{args.num_vec}_{loss_type}_{epoch}.png"
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="slateblue", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return


### Train ###
def main(logger, args, loss_type, dataloader, test_dataloader, True_j, vec, rolling, test_x):
    
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Define FNO3D
    model = FNO(
        in_channels=1,
        out_channels=8,
        decoder_layer_size=128,
        num_fno_layers=5,
        num_fno_modes=[2, 15, 15],
        padding=3,
        dimension=3,
        latent_channels=64
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)
    nx, ny = args.nx, args.ny

    # Create vjp_batch
    True_j = torch.tensor(True_j).float()
    True_j = True_j.reshape(-1, dataloader.batch_size, 5, args.num_vec, args.nx, args.ny)
    print("True J Before", True_j.shape, "After True J", True_j.shape) #([7, 100, 8, 64, 64]) -> ([idx, batchsize, 8, 64, 64])
    vec = torch.tensor(vec)
    print("vec", vec.shape)
    vec_batch = vec.reshape(-1, dataloader.batch_size, 5, args.num_vec, args.nx, args.ny)
    print("vec", vec_batch.shape)
    vec_batch = vec_batch.cuda().float()


    ### Training Loop ###
    elapsed_time_train, mse_diff, jac_diff_list, test_diff = [], [], [], []
    lowest_loss = float('inf')

    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for X, Y in dataloader:
            X, Y = X.cuda().float(), Y.cuda().float()
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)
            
            # MSE 
            if args.loss_type == "MSE":
                output = model(X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                if (epoch == 1) and (idx == 0):
                    plot_multiple_abs(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/true_sat_{epoch}.png")
                    plot_multiple_abs(Y[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/true_sat2_{epoch}.png")
                if (epoch % 10 == 0) and (idx == 0):
                    plot_multiple_abs(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/learned_sat_{epoch}.png")
                    plot_multiple_abs(output[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/learned_sat2_{epoch}.png")
                    plot_multiple_abs(abs(output[0]-Y[0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/diff_sat_{epoch}.png", "magma")
                    plot_multiple_abs(abs(output[1]-Y[1]).squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/diff_sat2_{epoch}.png", "magma")
            else:
            # GM
                print("idx for vjp and v: ", idx)
                # 1. update vjp and eigenvector
                target_vjp = True_j[idx].cuda()
                target_vjp = target_vjp.unsqueeze(1)  # Shape becomes [5, 1, 8, 15, 64, 64]
                target_vjp = target_vjp.permute(0, 1, 3, 2, 4, 5) # shape [5, 1, 15, 8, 64, 64]
                cur_vec_batch = vec_batch[idx] # 50 x 8 x 3 x 64 x 64
                
                # 2. compute MSE and GM loss term
                output, vjp_func = torch.func.vjp(model, X[:, :, 0].unsqueeze(dim=1))
                output = output.permute(0, 2, 1, 3, 4)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out_list = []
                for e in range(args.num_vec):
                    print("Compute regularization term ", e)
                    vjp_out_onevec = vjp_func(cur_vec_batch[:, :, e].unsqueeze(2))[0] # -> learned vjp
                    vjp_out_list.append(vjp_out_onevec)
                    vjp_out = torch.stack(vjp_out_list, dim=1)
                plot_multiple(vjp_out[0, : ,0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/learned_vjp_1st_sample.png", "seismic")
    
                if (epoch == 1) and (idx == 0):
                    plot_multiple(X[0,0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/K_{epoch}.png")
                    plot_multiple_abs(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/true_sat_{epoch}.png")
                    plot_multiple(cur_vec_batch[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/true_eigvec_{epoch}.png", cmap)
                    plot_multiple(target_vjp[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/true_vjp_{epoch}.png", cmap)
                if (epoch % 10 == 0) and (idx == 0):
                    plot_multiple_abs(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/learned_sat_{epoch}.png")
                    plot_multiple(vjp_out[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/learned_vjp_{epoch}.png", "seismic")
                    plot_multiple_abs(abs(output[0]-Y[0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/diff_sat_{epoch}.png", "magma")
                    plot_multiple_abs(abs(target_vjp[0, :, 0]-vjp_out[0, :, 0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/JAC/diff_vjp_{epoch}.png", "magma")

                jac_diff = criterion(target_vjp, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        # Save loss
        mse_diff.append(full_loss)
        if args.loss_type == "JAC":
            jac_diff_list.append(jac_misfit)
        # Save time
        end_time = time.time()  
        elapsed_time_train.append(end_time - start_time)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_dataloader:
                X_test = X_test.unsqueeze(1)
                Y_test = Y_test.unsqueeze(1)
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test)
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test) # relative error
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()

        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/GCS_vec_{args.num_vec}_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/GCS_vec_{args.num_vec}_{loss_type}_best.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.unsqueeze(1).cuda().float(), Y_test.unsqueeze(1).cuda().float()
            with torch.no_grad():
                Y_pred = model(X_test)
            plot_path = f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_vec_{args.num_vec}_{loss_type}_True.png"
            plot_multiple_abs(Y_test[0].squeeze().cpu(), plot_path)
            plot_multiple_abs(Y_pred[0].squeeze().detach().cpu(), f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_Pred.png")
            plot_multiple_abs(abs(Y_pred[0]-Y_test[0]).squeeze().detach().cpu(), f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_diff.png", "magma")
                

        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/GCS_vec_{args.num_vec}_{loss_type}_full.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/GCS_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])
    # Save the losses
    loss_data = [
        (mse_diff, 'mse_loss'),
        (jac_diff_list, 'jac_loss') if args.loss_type == "JAC" else (None, None),
        (test_diff, 'test_loss')
    ]
    for data, name in loss_data:
        if data:
            with open(f'../test_result/Loss/GCS_vec_{args.num_vec}_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/GCS_vec_{args.num_vec}_{loss_type}.png"

    # Remove the first few epochs (e.g., the first 5 epochs)
    start_epoch = 30
    mse_diff = mse_diff[start_epoch:]
    test_diff = test_diff[start_epoch:]
    jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")

    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="slateblue", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")

    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Final Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "Lowest Test Loss", str(lowest_loss))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model




if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--num_train", type=int, default=10)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--reg_param", type=float, default=20.0) # 0.1 -> 2
    parser.add_argument("--num_vec", type=int, default=8)

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"GCS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    cmap = LinearSegmentedColormap.from_list(
        "cmap_name",
        ["#0000FF", "white", "#FF0000"]
    )


    set_x, set_y, set_vjp, set_eig, set_rolling = [], [], [], [], []

    with h5py.File('../../../Diff_MultiPhysics/FNO-NF.jl/scripts/wise_perm_models_2000_new.jld2', 'r') as f:
        print("Keys: %s" % f.keys()) # Keys: <KeysViewHDF5 ['BroadK', 'K', 'NarrowK', 'phi']>

        K = f['BroadK'][:]
        # phi = f['phi'][:]
        print(len(K), len(K[0]), len(K[0][0])) # 256 x 512 x 2000
        set_x.append(K)
        set_x = set_x[0]

    ####
    # Transposing and Rescaling K. Should I multiply md?
    ####


    def resize_array(A, new_size):
        # A: original array
        # new_size: tuple of desired shape, e.g., (A.shape[0], dx, dy)
        
        # Determine the original grid (using 1-indexed coordinates as in the Julia code)
        orig_dims = A.shape
        grid = [np.linspace(1, d, d) for d in orig_dims]
        
        # Build the new coordinate grid for each dimension
        new_grid = [np.linspace(1, d, n) for d, n in zip(orig_dims, new_size)]
        
        # Create a meshgrid for the new coordinates.
        # Using 'ij' indexing so that the order of dimensions is preserved.
        mesh = np.meshgrid(*new_grid, indexing='ij')
        
        # Stack the grid arrays to get a (..., ndim) array of evaluation points.
        pts = np.stack(mesh, axis=-1)
        
        # Evaluate the interpolation using linear interpolation.
        # bounds_error=False with fill_value=None allows extrapolation.
        new_A = interpn(grid, A, pts, method='linear', bounds_error=False, fill_value=None)
        
        return new_A

    BroadK_rescaled = resize_array(BroadK, (BroadK.shape[0], 256, 256))

    # Read the each file s_idx: sample index
    for s_idx in range(1, args.num_train+args.num_test+1):
        with h5py.File(f'../data_generation/src/num_ev_{args.num_vec}/states_sample_{s_idx}.jld2', 'r') as f1, \
            h5py.File(f'../data_generation/src/num_ev_{args.num_vec}/FIM_eigvec_sample_{s_idx}.jld2', 'r') as f2, \
            h5py.File(f'../data_generation/src/num_ev_{args.num_vec}/FIM_vjp_sample_{s_idx}.jld2', 'r') as f3:

            states_refs = f1['single_stored_object'][:]  # Load the array of object references
            states_tensors = []
            # Loop over the references, dereference them, and convert to tensors
            for ref in states_refs:
                # Dereference the object reference
                state_data = f1[ref][:]
                
                # Convert the dereferenced data to a PyTorch tensor
                state_tensor = torch.tensor(state_data)
                states_tensors.append(state_tensor)
            
            eigvec = f2['single_stored_object'][:] # len: 8 x 20 x 64 x 64
            vjp = f3['single_stored_object'][:] # len: 8 x 20 x 4096
            # print(torch.tensor(eigvec).shape, torch.tensor(vjp).shape)

            # set_y.append(S) 
            set_y.append(torch.stack(states_tensors).reshape(8, 64, 64))
            set_vjp.append(torch.tensor(vjp).reshape(8, 20, 64, 64)[:, :args.num_vec]) 
            set_eig.append(torch.tensor(eigvec).reshape(8, 20, 64, 64)[:, :args.num_vec])


            # Plot every 200th element
            if s_idx % 200 == 0:
                plot_single_abs(set_x[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_permeability:{s_idx}.png")
                print(len(set_y), s_idx)
                plot_multiple_abs(set_y[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_saturation:{s_idx}.png")
                if args.loss_type == "JAC":
                    plot_multiple(torch.tensor(set_vjp[s_idx-1][0]), f"../plot/GCS_channel_vec_{args.num_vec}/data_vjp:{s_idx}.png", "seismic")
                    # plot_multiple(set_eig[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_eigvec:{s_idx}.png")


    print("len or:", len(set_x), len(set_x[0]), len(set_y), len(set_vjp))
    train_x_raw = torch.tensor(set_x[:args.num_train])
    train_y_raw = torch.stack(set_y[:args.num_train])
    test_x_raw = torch.tensor(set_x[args.num_train:args.num_train+args.num_test])
    test_y_raw = torch.stack(set_y[args.num_train:args.num_train+args.num_test])

    train_x_raw = train_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    train_x_raw = train_x_raw.repeat(1, 8, 1, 1)  # Now tensor is [25, 8, 64, 64]

    test_x_raw = test_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    test_x_raw = test_x_raw.repeat(1, 8, 1, 1)  # Now tensor is [25, 8, 64, 64]

    # normalize train_y_raw and train_vjp and set_eig
    def normalize_to_range(x, new_min=-1.0, new_max=1.0):
        """
        Normalize the tensor x to the range [new_min, new_max]
        """
        old_min = torch.min(x)
        old_max = torch.max(x)
        x_norm = (new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min
        return x_norm

    train_vjp = torch.stack(set_vjp[:args.num_train]).reshape(-1, 64, 64)
    print("vjp norm", torch.norm(train_vjp), torch.max(train_vjp))
    train_vjp = train_vjp / torch.norm(train_vjp)
    # train_vjp = train_vjp / (10**13)

    set_eig = torch.stack(set_eig[:args.num_train])
    print("len: ", len(train_vjp), len(set_eig))
    print("len:", len(train_x_raw), len(train_y_raw), len(test_x_raw), len(test_y_raw))

    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, train_vjp, set_eig, None, test_x_raw)