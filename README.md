# MobileNet v2 
A Keras 2 implementation of MobileNet V2.  

According to the paper:[Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

Currently only the network structure is defined, and the training function will be updated later.

## Requirement
- Python 3.5    
- Tensorflow-gpu 1.2.0  
- Keras 2.1.3


## MobileNet v2 and inverted residual block architectures
**MobileNet v2:**  

Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 X 3 kernels. The expansion factor t is always applied to the input size.

![MobileNetV2](/images/net.jpg)

**Residual Block Architectures:**

![residual block architectures](/images/stru.jpg)

**Architectures of this implementation with (224, 224, 3) inputs and 1000 output:**

![architectures](/images/MobileNetv2.png)

##Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

##Copyright
See [LICENSE](LICENSE) for details.


