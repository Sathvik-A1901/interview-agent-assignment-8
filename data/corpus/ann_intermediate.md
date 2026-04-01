# Artificial Neural Networks

## Forward Propagation
An artificial neural network processes input by passing values through layers of neurons. Each connection has a weight, and each neuron typically applies a weighted sum of its inputs plus a bias, then passes the result through a nonlinear activation function. Forward propagation is the process of computing the network output from input to output layer without updating weights. This feedforward structure allows hierarchical feature learning: early layers capture simple patterns while deeper layers combine them into richer representations used for classification or regression.

## Backpropagation and Gradients
Training adjusts weights using gradients of a loss function with respect to each parameter. Backpropagation applies the chain rule to propagate error signals from the output layer backward through the network, yielding partial derivatives efficiently. Rumelhart, Hinton, and Williams (1986) popularized this algorithm for multi-layer networks. Without backpropagation, optimizing deep networks would be computationally intractable. The learning rate scales each gradient step; too large a rate can diverge, while too small a rate slows convergence.

## Activation Functions
Activation functions introduce nonlinearity so stacked layers are not equivalent to a single linear map. Common choices include sigmoid and tanh, which squash outputs into bounded ranges, and ReLU, which outputs the input if positive and zero otherwise. ReLU often speeds training and mitigates vanishing gradients in deep feedforward networks compared to saturating sigmoids. The choice of activation affects gradient flow, sparsity of activations, and the kinds of functions the network can approximate.

## Loss Functions
The loss function measures how far predictions are from targets. For classification, cross-entropy compares predicted class probabilities to one-hot labels. For regression, mean squared error penalizes squared residual errors. The loss landscape is high-dimensional and non-convex; optimization finds local minima that generalize well when paired with regularization, data augmentation, and proper validation. The loss must be differentiable (or subdifferentiable) where backpropagation is applied.

## Vanishing Gradients in Deep ANNs
In very deep feedforward stacks with saturating activations, gradients can shrink exponentially as they propagate backward, slowing or freezing learning in early layers. Techniques such as better initialization, ReLU activations, residual connections, and layer normalization were developed to keep gradients at a healthy scale. This phenomenon is distinct from but related to the sequential vanishing gradient problem in recurrent networks, where repeated multiplication by transition Jacobians across time steps causes similar decay.
