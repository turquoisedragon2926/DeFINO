import torch
import torch.nn as nn
import pytorch_lightning as pl
from modulus.models.fno import FNO
from typing import Dict, Any, Optional, Tuple
import gc  # Add import for garbage collection


class NSModel(pl.LightningModule):
    """PyTorch Lightning module for NS saturation prediction with optional Jacobian regularization."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        decoder_layer_size: int = 128,
        num_fno_layers: int = 3,
        num_fno_modes: list = [2, 15, 15],
        padding: int = 3,
        dimension: int = 3,
        latent_channels: int = 64,
        loss_type: str = "MSE",
        reg_param: float = 0.01,
        scale_factor: float = 5500.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """
        Initialize the GCS model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            decoder_layer_size: Size of the decoder layer
            num_fno_layers: Number of FNO layers
            num_fno_modes: List of FNO modes for each dimension
            padding: Padding size
            dimension: Input dimension (3 for spacetime)
            latent_channels: Number of latent channels
            loss_type: Loss type to use (MSE, JAC)
            reg_param: Regularization parameter for Jacobian loss
            scale_factor: Scale factor for Jacobian loss
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create FNO model
        self.model = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_layer_size=decoder_layer_size,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            dimension=dimension,
            latent_channels=latent_channels
        )
        
        # Loss setup
        self.loss_type = loss_type
        self.reg_param = reg_param
        self.scale_factor = scale_factor
        
        # For tracking loss metrics
        self.train_rel_l2_loss = 0.0
        self.train_mse_loss = 0.0
        self.val_rel_l2_loss = 0.0
        self.jac_loss = 0.0
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def relative_l2_loss(self, true, pred):
        """Relative L2 loss."""
        return torch.norm(true - pred) / torch.norm(true)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch['x']
        y = batch['y']
        output = self.forward(x)
        
        # REL L2 loss
        rel_l2_loss = self.relative_l2_loss(y.squeeze(), output.squeeze())
        self.train_rel_l2_loss = rel_l2_loss.detach()
        
        # MSE loss
        mse_loss = torch.mean((y.squeeze() - output.squeeze()) ** 2)
        self.train_mse_loss = mse_loss.detach()
        
        # Total loss (default to REL L2)
        loss = rel_l2_loss
        
        # Log loss
        self.log('train_rel_l2_loss', rel_l2_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_mse_loss', mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        # TODO: Add Jacobian loss
            
        # Log total loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch['x']
        y = batch['y']
        
        # Forward pass
        y_pred = self.model(x)
        
        # REL L2 loss
        val_rel_l2_loss = self.relative_l2_loss(y.squeeze(), y_pred.squeeze())
        self.val_rel_l2_loss = val_rel_l2_loss.detach()
        
        # Log validation loss
        self.log('val_rel_l2_loss', val_rel_l2_loss, prog_bar=True, on_epoch=True)
        
        return {'val_loss': val_rel_l2_loss}
        
    def _clear_memory(self):
        """Clear CUDA memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Clear memory at the end of each epoch
        self._clear_memory()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Clear memory at the end of each validation epoch
        self._clear_memory()
    
    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        return optimizer
