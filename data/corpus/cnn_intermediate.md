# Convolutional Neural Networks

## The Convolution Operation
Convolutional layers slide learnable filters across spatial dimensions of the input, computing local dot products and producing feature maps. Sharing weights across locations drastically reduces parameters compared to fully connected layers and encodes translation equivariance: the same pattern is detected regardless of position. For images, convolutions exploit spatial locality and hierarchical structure. Stride controls downsampling within the convolution, and padding preserves spatial size when desired.

## Pooling and Downsampling
Pooling layers aggregate small neighborhoods (often with max or average) to reduce spatial resolution and add a degree of translation invariance. Max pooling passes the strongest activation in each window, emphasizing salient features. Downsampling can also be achieved through strided convolutions, which some modern architectures prefer because they remain fully learnable. Excessive pooling can discard fine spatial detail needed for dense prediction tasks such as segmentation.

## Feature Hierarchies
Early convolutional layers tend to respond to edges, textures, and simple patterns, while deeper layers encode parts and object-level concepts. This hierarchical representation emerges from stacking convolutions and nonlinearities without explicit hand-designed features. LeCun et al. (1998) demonstrated this paradigm with LeNet on digit recognition. Deep CNNs learn representations suited to the training objective, which is why pretraining and transfer learning are effective across related visual tasks.

## LeNet and Classical CNN Design
LeNet combined convolutional layers, subsampling, and fully connected layers for handwritten character recognition. It established the blueprint of convolutions for local features followed by spatial aggregation. Although shallow by modern standards, LeNet showed that end-to-end learning of convolutional features outperformed many engineered pipelines of its era. The design principles remain relevant: local connectivity, parameter sharing, and increasing receptive field depth.

## AlexNet and Deep CNN Scale
Krizhevsky et al. (2012) scaled CNNs with GPU training, ReLU activations, dropout regularization, and data augmentation on ImageNet. AlexNet dramatically reduced error rates on large-scale image classification and reignited interest in deep learning for vision. The architecture used stacked convolutions, pooling, and large fully connected layers. Its success demonstrated that depth, compute, and big data together unlock performance gains that shallow models cannot match.
