import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassAccuracy
import torcheval.metrics

# Define a general LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, model_net, optimizer = None, config_optimizer = None, schedule = False):
       
        super().__init__()
        
        print('Network initialized')
        
        self.net = model_net

        """
        From the website of torchmetrics:https://lightning.ai/docs/torchmetrics/stable/pages/overview.html
        Better to always try to reuse the same instance of a metric instead of initializing a new one.
        Calling the reset method returns the metric to its initial state, and can therefore be used to reuse the same instance. 
        However, we still highly recommend to use different instances from training, validation and testing.


        ----

        To save a metric:
        EXAMPLE:

        import torch
        from torchmetrics.classification import MulticlassAccuracy

        metric = MulticlassAccuracy(num_classes=5).to("cuda")
        metric.persistent(True)
        metric.update(torch.randint(5, (100,)).cuda(), torch.randint(5, (100,)).cuda())
        torch.save(metric.state_dict(), "metric.pth")

        metric2 = MulticlassAccuracy(num_classes=5).to("cpu")
        metric2.load_state_dict(torch.load("metric.pth", map_location="cpu"))

        # These will match, but be on different devices
        print(metric.metric_state)
        print(metric2.metric_state)

        ----

        Since I log all the metrics, the only one that I have to save manually is the confusion matrix.
        
        """
        # Adding .persistent(True) only for the confusion matrix, since I log the others.
        self.accuracy_train = MulticlassAccuracy(num_classes=8)
        self.accuracy_val   = MulticlassAccuracy(num_classes=8)
        self.accuracy_test  = MulticlassAccuracy(num_classes=8)

        self.confusion_matrix_train = MulticlassConfusionMatrix(num_classes=8)
        self.confusion_matrix_val   = MulticlassConfusionMatrix(num_classes=8)
        self.confusion_matrix_test  = MulticlassConfusionMatrix(num_classes=8)

        self.confusion_matrix_train.persistent(True)
        self.confusion_matrix_val.persistent(True)

        self.f1_score_train = MulticlassF1Score(num_classes=8, average='macro')
        self.f1_score_val   = MulticlassF1Score(num_classes=8, average='macro')
        self.f1_score_test  = MulticlassF1Score(num_classes=8, average='macro')


        """
        self.top2_accuracy_train = MulticlassAccuracy(num_classes=8, k=2)
        self.top2_accuracy_val = MulticlassAccuracy(num_classes=8, k=2)
        self.top2_accuracy_test = MulticlassAccuracy(num_classes=8, k=2)
        
        """

       
    # If no optimizer is passed, the default optimizer is Adam
        



        #try:
        #    self.optimizer = Adam(self.net.parameters(),
        #                               lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
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

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch     = batch[0]
        label_batch = batch[1]
        out         = self.net(x_batch)
        loss        = F.cross_entropy(out, label_batch) 

        """
        To see how to use the metrics, check the following link:
        https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html

        Trying not to mix self.log with .update and .compute

        """
        
        #Estimation of model accuracy
        out_argmax = out.argmax(dim=1)
        label_argmax = label_batch.argmax(dim=1)

        self.log("train_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True)
        self.log("train_acc", self.accuracy_train(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("train_f1_score",self.f1_score_train(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("train_top2_acc",self.top2_accuracy_train(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_train.update(out_argmax, label_argmax)
        
        return loss
    
    def on_train_epoch_end(self):
        # Save the confusion matrix
        torch.save(self.confusion_matrix_train.state_dict(), "confusion_matrix_train.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_train.compute()
        print(cm)
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
        self.log("val_acc", self.accuracy_val(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("val_f1_score",self.f1_score_val(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("val_top2_acc",self.top2_accuracy_val(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_val.update(out_argmax, label_argmax)


    
    def on_validation_epoch_end(self):
        
        # Save the confusion matrix
        torch.save(self.confusion_matrix_val.state_dict(), "confusion_matrix_val.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_val.compute()
        print(cm)
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

        #Further metrics can be precisione and recall...

        self.log("test_loss", loss.item(), prog_bar=True,on_step=False,on_epoch=True)
        self.log("test_acc", self.accuracy_test(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("test_f1_score",self.f1_score_test(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.log("test_top2_acc",self.top2_accuracy_test(out_argmax, label_argmax), prog_bar=True,on_step=False,on_epoch=True)
        self.confusion_matrix_test.update(out_argmax, label_argmax)

    def on_test_epoch_end(self):

        # Save the confusion matrix
        torch.save(self.confusion_matrix_test.state_dict(), "confusion_matrix_test.pth")
        print("computing confusion matrix")
        cm = self.confusion_matrix_test.compute()
        print(cm)
        print("Resetting confusion matrix")
        self.confusion_matrix_test.reset()

       
    def configure_optimizers(self):
        # This function is called by Lightning to configure the optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

        #print("Using StepLR with step_size = {} and gamma = {}".format(self.lr_step, self.lr_gamma))

        if self.schedule:
            print("Using scheduler")
            return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}
            
        else:
            print("Not using scheduler")   
            return self.optimizer



# Adding BatchNorm to avoid overfitting
class NNET1D(nn.Module):
        
    def __init__(self):
        super(NNET1D, self).__init__()
        
        
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
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 




# Start by removing stuff that requires an experiment, like Dropout or BatchNorm
class NNET2D(nn.Module):
        
    def __init__(self, initialisation="xavier"):
        super(NNET2D, self).__init__()
        
        
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
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        
        
    def forward(self,x):
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1)) # Reshape x to fit in linear layers. Equivalent to F.Flatten
        return x  







class MixNet(nn.Module):
    def __init__(self, conv_block1D, conv_block2D):
        super(MixNet, self).__init__()
        self.conv_block1D = conv_block1D
        self.conv_block2D = conv_block2D

        self.classifier = nn.Sequential(
            nn.Linear(1024+1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),   
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 8),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        
        

    def forward(self, x):
        audio = x[0]
        spectrogram = x[1]
        
        conv2d = self.conv_block2D(spectrogram)
        max_pool = F.max_pool2d(conv2d, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(conv2d, kernel_size=(125,1))
        cat2d = torch.cat([max_pool,avg_pool],dim=1)
        cat2d = cat2d.view(cat2d.size(0), -1) # cat2d shape torch.Size([1, 512])
        
        conv1d = self.conv_block1D(audio)
        max_pool = F.max_pool1d(conv1d, kernel_size=125)
        avg_pool = F.avg_pool1d(conv1d, kernel_size=125)
        cat1d = torch.cat([max_pool,avg_pool],dim=1)
        cat1d = cat1d.view(cat1d.size(0), -1) # cat1d dim = torch.Size([batch_size, 2048])

        # Concatanate the two outputs and pass it to the classifier
        # cat1d dim = torch.Size([batch_size, 2048])
        # cat2d dim = torch.Size([batch_size, 512])
        x = torch.cat([cat1d, cat2d], dim=1) 
        x = self.classifier(x)
        return x
