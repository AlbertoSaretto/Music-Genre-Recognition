import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassAccuracy
import torcheval.metrics



########################################################################################
# General LightningModule
########################################################################################


# Define a general LightningModule (nn.Module subclass)
class LitNet(pl.LightningModule):
    
    def __init__(self, model_net, optimizer = None, config_optimizer = None, schedule = False):
       
        super().__init__()
        
        print('Network initialized')
        
        self.net = model_net

        self.confusion_matrix_train = MulticlassConfusionMatrix(num_classes=8)
        self.confusion_matrix_val   = MulticlassConfusionMatrix(num_classes=8)
        self.confusion_matrix_test  = MulticlassConfusionMatrix(num_classes=8)

        self.confusion_matrix_train.persistent(True)
        self.confusion_matrix_val.persistent(True)

        self.f1_score_train = MulticlassF1Score(num_classes=8, average='macro')
        self.f1_score_val   = MulticlassF1Score(num_classes=8, average='macro')
        self.f1_score_test  = MulticlassF1Score(num_classes=8, average='macro')

        
        if optimizer is None:
                print("Using default optimizer parameters")
                self.optimizer = Adam(self.net.parameters(), lr = 1e-5)
        else:
            
            print("Using optimizer passed as argument")
            self.optimizer = optimizer

        try:
            self.schedule = schedule
            self.lr_step = config_optimizer["lr_step"]
            self.lr_gamma = config_optimizer["lr_gamma"]

            print("Using lr_step and lr_gamma from config_optimizer")

        except:
            self.schedule = False
            self.lr_step = 1
            self.lr_gamma = 0.0


    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch     = batch[0]
        label_batch = batch[1]
        out         = self.net(x_batch)
        loss        = F.cross_entropy(out, label_batch) 
        
        #Estimation of model accuracy
        out_argmax = out.argmax(dim=1)
        label_argmax = label_batch.argmax(dim=1)

        self.log("train_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True)
        self.log("train_f1_score",self.f1_score_train(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_train.update(out_argmax, label_argmax)
        
        return loss
    
    def on_train_epoch_end(self):
        # Save the confusion matrix
        torch.save(self.confusion_matrix_train.state_dict(), "confusion_matrix_train.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_train.compute()
        print(cm)

        # Compute accuracy
        correct_predictions = torch.diag(cm).sum().item()
        total_predictions = cm.sum().item()
        accuracy = correct_predictions / total_predictions
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        print("Resetting confusion matrix")
        self.confusion_matrix_train.reset()



    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.

        x_batch     = batch[0]
        label_batch = batch[1]

        out  = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        #Estimation of model accuracy
        out_argmax = out.argmax(dim=1)
        label_argmax = label_batch.argmax(dim=1)

        self.log("val_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True)
        self.log("val_f1_score",self.f1_score_val(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_val.update(out_argmax, label_argmax)


    
    def on_validation_epoch_end(self):
        
        # Save the confusion matrix
        torch.save(self.confusion_matrix_val.state_dict(), "confusion_matrix_val.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_val.compute()
        print(cm)

        # Compute accuracy
        correct_predictions = torch.diag(cm).sum().item()
        total_predictions = cm.sum().item()
        accuracy = correct_predictions / total_predictions
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        print("Resetting confusion matrix")
        self.confusion_matrix_val.reset()

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        #Estimation of model accuracy
        out_argmax = out.argmax(dim=1)
        label_argmax = label_batch.argmax(dim=1)

        self.log("test_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True)
        self.log("test_f1_score",self.f1_score_test(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_test.update(out_argmax, label_argmax)

    def on_test_epoch_end(self):

        # Save the confusion matrix
        torch.save(self.confusion_matrix_test.state_dict(), "confusion_matrix_test.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_test.compute()
        print(cm)

        # Compute accuracy
        correct_predictions = torch.diag(cm).sum().item()
        total_predictions = cm.sum().item()
        accuracy = correct_predictions / total_predictions
        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        print("Resetting confusion matrix")
        self.confusion_matrix_test.reset()

       
    def configure_optimizers(self):
        # This function is called by Lightning to configure the optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

        if self.schedule:
            print("Using scheduler")
            return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}
            
        else:
            print("Not using scheduler")   
            return self.optimizer



########################################################################################
# 1D network
########################################################################################


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
    

class NNET1D(nn.Module):
        
    def __init__(self):
        super(NNET1D, self).__init__()
        
        self.conv_block = CONV1D()
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
            nn.Linear(64, 8),
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
        

    def forward(self, x):

        x = self.conv_block(x)
        max_pool = F.max_pool1d(x, kernel_size=64)
        avg_pool = F.avg_pool1d(x, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x 



########################################################################################
# 2D network
########################################################################################


class CONV2D(nn.Module):
    def __init__(self, initialisation="xavier"):
        super(CONV2D, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,20)),
            nn.BatchNorm2d(256),
            nn.ReLU()     
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256)       
        )

        self.apply(self._init_weights)

        self.relu = nn.ReLU()

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
               
    def forward(self,x):
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = self.relu(c1 + c3)
        return x



class NNET2D(nn.Module):
           
    def __init__(self, initialisation="xavier"):
        super(NNET2D, self).__init__()
        
        self.conv_block = CONV2D()
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)        
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            
        
        
    def forward(self,x):
        x = self.conv_block(x)
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1))
        return x
    

########################################################################################
# MixNet network
########################################################################################

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

########################################################################################