from mgr.models import NNET1D, LitNet
from mgr.utils_mgr import import_and_preprocess_data, create_dataloaders
import pytorch_lightning as pl

def main(max_epochs,model):
    pl.seed_everything(0)

   
      
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )


    # I think that Trainer automatically takes last checkpoint.
    trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=5, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], ) # profiler="simple" remember to add this and make fun plots
    
    #hyperparameters = load_optuna()
    #model = LitNet(hyperparameters) 
    
    # Load model weights from checkpoint
    #CKPT_PATH = "./lightning_logs/version_16/checkpoints/epoch=19-step=2000.ckpt"
    #checkpoint = torch.load(CKPT_PATH)
    #model.load_state_dict(checkpoint['state_dict'])
    

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(PATH_DATA="/home/diego/Music-Genre-Recognition/data/",num_workers=8)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)

if __name__ == "__main__":
    main(max_epochs=1,model = LitNet(NNET1D()))

