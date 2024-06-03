# Music Genre Recognition with Convolutional Neural Networks

We present a deep learning approach to solve the music genre classification task using the [FMA dataset](https://github.com/mdeff/fma).


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
