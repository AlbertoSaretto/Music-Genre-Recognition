# Music Genre Recognition with Convolutional Neural Networks

We present a deep learning approach to solve the music genre classification task using the [FMA dataset](https://github.com/mdeff/fma). Three different architectures have been developed in PyTorch, exploiting different representation of audio signals, namely 1D audio clips, 2D spectrograms and a mix of the two. 

The models are trained on 8 differenet music genre classes and they reach the following final scores on the test set:

| Model | Accuracy | Cross-Entropy Loss | F1 Score |
|-------|----------|--------------------| ---------|
| CNN1D | 47%      | 1.73               | 0.33 |
| CNN2D | 51%      | 1.70               | 0.38 |
| MixNet| 55%      | 1.65               | 0.35 |




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
This arichtecture was created by combining the convolutional blocks of the previous networks. It thus extracts and exploits information from both 1D audio signals and 2D spectrograms.

<p align="center">
  <img src="imgs/cnnmix_scheme.jpg" alt="MixNet Architecture" width="400" />
</p>
