"""
Nota di Diego 3-3-2024
Io discarderei questo script e lo rifarei usando main_train (vedi exploring_transformations.py)

"""


import warnings
#disable warnings
warnings.filterwarnings("ignore")

from mgr.models import NNET2D, LitNet
from mgr.utils_mgr import import_and_preprocess_data, create_dataloaders
import pytorch_lightning as pl
import torchvision.transforms.v2 as v2
import torch
import os




def main(max_epochs,model):
    pl.seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # release all unoccupied cached memory
        torch.cuda.empty_cache()
        # printo GPU info
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print('{} {} GPU available'.format(str(device_count), str(device_name)))

      
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )

    # Set the trainer's device to GPU if available
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[early_stop_callback],
        devices ="auto", # Diego: questo mi dà problemi
        accelerator='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    transforms = v2.Compose([v2.ToTensor()])
    
    #hyperparameters = load_optuna()
    #model = LitNet(hyperparameters) 
    
    # Load model weights from checkpoint
    #CKPT_PATH = "./lightning_logs/version_16/checkpoints/epoch=19-step=2000.ckpt"
    #checkpoint = torch.load(CKPT_PATH)
    #model.load_state_dict(checkpoint['state_dict'])
    

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(PATH_DATA="../data/", transforms=transforms,num_workers=8,batch_size=64, net_type = "2D")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)

if __name__ == "__main__":
    main(max_epochs=5,model = LitNet(NNET2D(), lr=1e-3))
