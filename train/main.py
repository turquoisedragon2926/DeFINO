import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from callbacks import SaturationVisualizationCallback, JacobianVisualizationCallback
from utils import (
    load_config, save_config, setup_logging, setup_neptune_logging,
    create_directories, log_config_to_file, set_seed, get_dataset, get_model
)

try:
    from pytorch_lightning.loggers import NeptuneLogger
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


def main():
    """Main function to run the training process."""
    parser = argparse.ArgumentParser(description='GCS FNO Training with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/mse.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directories
    directories = create_directories(config)
    
    # Set up logging
    log_file, logger = setup_logging(
        log_dir=directories['log_dir'],
        experiment_name=directories['experiment']
    )
    
    # Log configuration
    log_config_to_file(config, logger)
    
    # Save configuration to experiment directory
    save_config(config, os.path.join(directories['output_dir'], 'config.yaml'))
    
    # Set random seed for reproducibility
    set_seed(config.experiment.seed)
    
    # Set up Neptune logging if enabled
    neptune_run = setup_neptune_logging(config)
        
    data_loader = get_dataset(config.experiment.dataset_type, config.data_settings)
    model = get_model(config.experiment.model_type, config.model_settings)
    
    train_loader = data_loader.get_dataloader(offset=0, limit=config.training_settings.num_train, shuffle=False) # TODO: figure out how to set to true & have same plotting in callback
    test_loader = data_loader.get_dataloader(offset=config.training_settings.num_train, limit=config.training_settings.num_test, shuffle=False)

    # Set up loggers
    loggers = []
    
    # # TensorBoard logger
    # tb_logger = TensorBoardLogger(
    #     save_dir=directories['log_dir'],
    #     name="tensorboard",
    #     version=directories['experiment']
    # )
    # loggers.append(tb_logger)
    
    # Neptune logger
    if NEPTUNE_AVAILABLE and config.neptune.enabled:
        neptune_logger = NeptuneLogger(
            run=neptune_run,
            log_model_checkpoints=config.training_settings.log_checkpoint
        )
        loggers.append(neptune_logger)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=directories['checkpoint_dir'],
        filename=f"{config.experiment.name}_{{epoch:03d}}_{{val_rel_l2_loss:.4f}}",
        monitor="val_rel_l2_loss",
        save_top_k=config.training_settings.save_top_k,
        mode="min",
        save_last=False,
        every_n_epochs=config.training_settings.checkpoint_interval
    )
    if config.training_settings.enable_checkpointing:
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_rel_l2_loss",
        patience=20,
        min_delta=1e-4,
        mode="min"
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # TODO: Make callbacks dynamic based on config / dataset type

    # Create trainer and set checkpointing to false
    trainer = pl.Trainer(
        max_epochs=config.training_settings.num_epoch,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=True,
        check_val_every_n_epoch=config.training_settings.check_val_every_n_epoch,
        accumulate_grad_batches=config.training_settings.accum_steps,
        enable_checkpointing=config.training_settings.enable_checkpointing,
    )
    
    # Start training
    logger.info(f"Starting training...")
    # trainer.fit(model, train_loader, test_loader)
    
    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")
    
    # Close Neptune run if it was initialized
    if neptune_run:
        neptune_run.stop()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()
