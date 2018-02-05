# MobileNet v2 
A Python 3 and Keras 2 implementation of MobileNet V2 and provide train method.  

According to the paper: [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381).


## Requirement
- OpenCV 3.4
- Python 3.5    
- Tensorflow-gpu 1.2.0  
- Keras 2.1.3


## MobileNet v2 and inverted residual block architectures

**MobileNet v2:**  

Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 X 3 kernels. The expansion factor t is always applied to the input size.

![MobileNetV2](/images/net.jpg)

**Bottleneck Architectures:**

![residual block architectures](/images/stru.jpg)


## Train the model

The recommended size of the image in the paper is 224 * 224. The ```data\convert.py``` file provide a demo of resize cifar-100 dataset to this size.

**The dataset folder structure is as follows:**

	| - data/
		| - train/
	  		| - class 0/
				| - image.jpg
					....
			| - class 1/
			  ....
			| - class n/
		| - validation/
	  		| - class 0/
			| - class 1/
			  ....
			| - class n/

**Run command below to train the model:**

```
python train.py --classes num_classes --batch batch_size --epochs epochs --size image_size
```

The ```.h5``` weight file was saved at model folder. If you want to do fine tune the trained model, you can run the following command. However, it should be noted that the size of the input image should be consistent with the original model.

```
python train.py --classes num_classes --batch batch_size --epochs epochs --size image_size --weights weights_path --tclasses pre_classes
```

**Parameter explanation**

- --classes, The number of classes of dataset.  
- --size,    The image size of train sample.  
- --batch,   The number of train samples per batch.  
- --epochs,  The number of train iterations.  
- --weights, Fine tune with other weights.  
- --tclasses, The number of classes of pre-trained model.

## Experiment

Due to the limited computational resources, we used cifar-100 dataset to test the model.
	
	device: Tesla K80
	dataset: cifar-100
	optimizer: Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
	batch_szie: 128 

These are the details for the CIFAR-100 experiment. Although the network did not completely converge, still achieved good accuracy.

| Metrics | Loss | Top-1 Accuracy | Top-5 Accuracy |
| ------- |------| :------------: | :------------: |
| cifar-100 | 0.195 | 94.42% | 99.82% |

![evaluate](/images/eva.png)
## Reference

	@article{MobileNetv2,  
	  title={Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentatio},  
	  author={Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen},
	  journal={arXiv preprint arXiv:1801.04381},
	  year={2018}
	}


## Copyright
See [LICENSE](LICENSE) for details.


