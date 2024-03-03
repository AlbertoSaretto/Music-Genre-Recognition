"""
ME LO SEGNO QUA PERCHé NON SAPRE IDOVE ALTRO SEGNARLO

from argparse import ArgumentParser


def main(hparams):
    model = LightningModule()
    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)

"""


from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import main_train
import torch
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import audiomentations as audio

"""
https://pytorch.org/audio/stable/transforms.html

https://github.com/iver56/audiomentations/tree/main
"""

# 1D transformations 

"""
Fare esperimenti per capire se e quali trasformazioni sono utili per migliorare accuracy
Puoi trovare trasformazioni per audio 1D in due librerie: librosa e audiomentations
Trasformazioni 2D con torchaudio.

"""

"""
Qui mostro solo come io approccerei l'uso delle trasformazioni in un generico train 1D.
2D è simile, ma con trasformazioni diverse.
Vedi explore_transformations.ipynb


"""

if __name__ == "__main__":
    """
    # For more trasformations see notebook exploring_trasformations.ipynb
    train_audio_transform = audio.Compose([   
    
        audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        audio.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        
    ])
     
    main_train(max_epochs=10,
               model_net = LitNet(NNET1D()),
               net_type='1D',
               transforms=train_audio_transform,
               PATH_DATA="../data/",
               batch_size=32,
              )
    
    """
    transf = v2.Compose([v2.ToTensor()])

    main_train(max_epochs=1,
               model_net = LitNet(NNET2D()),
               net_type='2D',
               transforms=transf,
               PATH_DATA="../data/",
               batch_size=64,
               lr = .01
              )
    



"""
Cose da fare:
trasformazioni per 1D e 2D. Fare esperimenti per vedere se trasformazioni migliorano o peggiorano accuracy
Plot per verificarne la correttezza

"""