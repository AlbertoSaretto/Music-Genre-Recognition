from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import main_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import audiomentations as audio
import os



#OPTUNA RESULTS:
# weight decay: 0.000572


#Class to apply random transformations to the data
class RandomApply:
    def __init__(self, transform, prob):
        self.transform = transform
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            return self.transform(x)
        return x
    
class GaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        #return tensor + absolute value of noise
        return tensor + torch.abs(torch.randn(tensor.size()) * self.std + self.mean)





#////////////////////////////////////////////////////////////////////////////////////
#Experiment with convolutional block
#////////////////////////////////////////////////////////////////////////////////////


class CONV2D(nn.Module):
    def __init__(self, initialisation="xavier"):
        super(CONV2D, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,20)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            
        
        
    def forward(self,x):
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        #x = c1 + c3
        return c3
    

class CONV1D(nn.Module):

    def __init__(self):
        super(CONV1D, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
         
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
           
        )
        
    def forward(self, x):

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        return c4
        

    

    



#////////////////////////////////////////////////////////////////////////////////////








class MixNet(nn.Module):
    def __init__(self, conv_block1D, conv_block2D):

        super(MixNet, self).__init__()

        self.conv_block1D = conv_block1D
        self.conv_block2D = conv_block2D

        self.classifier = nn.Sequential(
            nn.Linear(1024+1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),   
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 8),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        
        

    def forward(self, x):
        audio = x[0]
        spectrogram   = x[1]
        
        # 2D BLOCK
        conv2d = self.conv_block2D(spectrogram)
        max_pool = F.max_pool2d(conv2d, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(conv2d, kernel_size=(125,1))
        cat2d = torch.cat([max_pool,avg_pool],dim=1)
        cat2d = torch.flatten(cat2d, start_dim=1)
        
        # 1D BLOCK
        conv1d = self.conv_block1D(audio)
        max_pool = F.max_pool1d(conv1d, kernel_size=64)
        avg_pool = F.avg_pool1d(conv1d, kernel_size=64)
        cat1d = torch.cat([max_pool,avg_pool],dim=1)
        cat1d = torch.flatten(cat1d, start_dim=1) 
        
        # Concatanate the two outputs
        x = torch.cat([cat1d, cat2d], dim=1) 
        x = self.classifier(x)
        return x
 


#E il  modo più semplice e pulito per farlo? Pare di si senza ridefinire tutto... CONTROLLA
def build_convolutional_blocks(nnet1d, nnet2d):
    
    # Get all convolutional layers from nnet2d
    conv_layers_2d = [layer for layer in nnet2d.modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU)]

    # Build a new convolutional layer
    conv_block2D = nn.Sequential(*conv_layers_2d[:9]) # [:9] to remove redundat ReLU layers taken by mistake from fc layers
        
    # Get all convolutional layers from nnet1d
    conv_layers_1d = [layer for layer in nnet1d.modules() if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.MaxPool1d)]

    # Build a new convolutional layer
    conv_block1D = nn.Sequential(*conv_layers_1d[:15])


    return conv_block1D, conv_block2D




        



if __name__ == "__main__":

    train_transform = { '2D':v2.Compose([
                            v2.ToTensor(),
                            RandomApply(FrequencyMasking(freq_mask_param=30), prob=0.5),     #Time and Freqeuncy are inverted bacause of the data are transposed
                            RandomApply(TimeMasking(time_mask_param=2), prob=0.5),
                            #Add Gaussian noise on spectrogram with v2
                            RandomApply(GaussianNoise(std = 0.015), prob=0.5),
                            ]),

                        '1D': audio.Compose([   
                            # add gaussian noise to the samples
                            audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                            # change the speed or duration of the signal without changing the pitch
                            #audio.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                            # pitch shift the sound up or down without changing the tempo
                            #audio.PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
                            # adds silence
                            audio.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
                            ])
    }



    eval_transform = {'2D':v2.Compose([v2.ToTensor(),             
                                      ]),
                      '1D': None           
    }


    # Load model weights from checkpoint

    '''

    CKPT_PATH_1D = "../CNN1D/lightning_logs/full_train_BN_transform/checkpoints/epoch=39-step=16000.ckpt"
    CKPT_PATH_2D = "../CNN2D/lightning_logs/all_trans5_champion/checkpoints/epoch=75-step=7600.ckpt"
    
    print("Loading models...")

    weights1D = torch.load(CKPT_PATH_1D)['state_dict']
    weights2D = torch.load(CKPT_PATH_2D,map_location=torch.device('cpu'))['state_dict']
    nnet1d = LitNet(NNET1D())
    nnet2d = LitNet(NNET2D())

    nnet1d.load_state_dict(weights1D)
    nnet2d.load_state_dict(weights2D)
   
    #nnet1d = LitNet.load_from_checkpoint(checkpoint_path=CKPT_PATH_1D).eval()
    #nnet2d = LitNet.load_from_checkpoint(checkpoint_path=CKPT_PATH_2D).eval()

    """    # Freeze the weights if you don't want to train the CNN part
    for param in nnet1d.parameters():
        param.requires_grad = False
        
    for param in nnet2d.parameters():
        param.requires_grad = False
    """
    print("Models loaded.")
    # Build convolutional blocks
    conv_block1D, conv_block2D = build_convolutional_blocks(nnet1d, nnet2d)
    '''

    # Build the model

    # Uncomment the following to load Optuna hyperparameters
    #hyperparameters = load_optuna("./trialv2.pickle")
    #model = LitNet(hyperparameters)
    
    # Comment this if you want to load params with Optuna
    model_net = MixNet(CONV1D(), CONV2D())
   
    
    config_optimizer = {'lr': 5e-5,
              'lr_step': 100,
              'lr_gamma': 0,
              'weight_decay': 0,
              }
    
    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 32,
                    'num_workers': 4,
                    'patience': 20,
                    'net_type':'Mix',
                    'mfcc': True,
                    'normalize': True,
                    'schedule': False
                    }
    



    main_train(model_net = model_net,
                train_transforms=train_transform,
                eval_transforms= eval_transform,
                PATH_DATA="../data/", 
                config_optimizer=config_optimizer,
                config_train=config_train,
                )
    

##tensorboard --logdir=lightning_logs/ 
# to visualize logs