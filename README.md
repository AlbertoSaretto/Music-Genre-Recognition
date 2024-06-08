# Music Genre Recognition with Convolutional Neural Networks

We present a deep learning approach to solve the music genre classification task using the [FMA dataset](https://github.com/mdeff/fma). Three different architectures have been developed in PyTorch, exploiting different representation of audio signals, namely 1D audio clips, 2D spectrograms and a mix of the two. 

The models are trained on 8 different music genre classes and they reach the following final scores on the test set:

| Model | Accuracy | Cross-Entropy Loss | F1 Score |
|-------|----------|--------------------| ---------|
| CNN1D | 47%      | 1.73               | 0.33 |
| CNN2D | 51%      | 1.70               | 0.38 |
| MixNet| 55%      | 1.65               | 0.35 |

You can find a detailed report in the report.pdf (to be added soon).


## 1D CNN
Convolutional neural network using 1D audio clips as input data.

<p align="center">
  <img src="imgs/cnn1D_scheme.jpg" alt="1D CNN Architecture" width="400" />
</p>


## 2D CNN 
Residual Convolutional neural network using 2D spectrograms as input data.
<p align="center">
  <img src="imgs/cnn2D_scheme.jpg" alt="2D CNN Architecture" width="400" />
</p>

## MixNet 
This network exploits information from both 1D audio signals and 2D spectrograms.
The CNN blocks are the 1D CNN and 2D CNN presented above. 


<p align="center">
  <img src="imgs/cnnmix_scheme.jpg" alt="MixNet Architecture" width="400" />
</p>


## How to use
In this repository you will find the following folders.

* `CNN1D`: containing the files related to the neural network working with 1D audio signals.
* `CNN2D`: containing the files related to the neural network working with 2D audio signals.
* `MixNet`: containing the files related to the neural network working with both 1D and 2D audio signals.
* `MGR`: the python package needed to run all the files in this repository.

Begin by installing the MGR package. Open a terminal and go inside the MGR directory. Run `pip install .`. Remember to install the package again anytime you modify a file inside of it, otherwise the change won't be registered. Also, we advise to create a new environment starting from the file `mgr_env.yml` (to be added soon).

In each folder, you can find two .py files: one is named after the network that is uses (e.g. `cnn1d.py`), the other is `hypertune.py` and it is used for the fine-tuning of the models. You will also find a `lightning_logs` folder, that is used by Lightining Pytorch to store useful data such as checkpoints. 
