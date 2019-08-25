# Basic terms
## Perceptron
Perceptron, also called linear binary classifier, is a single layer neural network that consists of 4 parts:
 - Input value (Assume a vector of attributes or smth)
 - Weigths & Bias
 - Net sum
 - Activation function

The perceptron takes in a vector of attributes (marked as x), multiplies each of them by it's corresponding weight (marked as w), and then sums all into a single value.
$$
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 
\end{bmatrix}
\dot{}
\begin{bmatrix}
w_1 \\
w_2 \\
w_3 
\end{bmatrix}
=
\begin{bmatrix}
x_1 * w_1 \\
x_2 * w_2\\
x_3 * w_3
\end{bmatrix} => x_1*w_1 + x_2*w_2 + x_3*w_3 = z
$$
An activation function (step function) then takes in that weighted sum (marked as z), and produces a binary output based on it's characteristics.

Weight of a node specify it's significance, and a bias allows to shift the activation (step) function, up or down.
## Neuron
The basic principal described above, applies here. The neuron takes in a vector of attributes and multiplies it by it's coresponding weights. The main difference lies in an activation function. Whereas, a perceptron uses a step function, a neuron usually uses some nonlinear one. Therefore, a neuron can be as descriptive as a perceptron, but not the other way around.

Small note: ***Logit*** is a raw (non-normalized) prediction produced by a network, which is then feeded into a normalization function (eg. softmax).

### Sigmoid neuron
Neuron with a sigmoid activation function:
$$
f(z) = \frac{1}{1 + e^{-z}}, f(z) ∈ [0, 1]
$$

Plot of a sigmoid function shows an s-shaped curve:
![](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F0*WYB0K0zk1MiIB6xp.png&f=1?raw=true)
### Tanh
Neuron with a tahn activation function:
$$
f(z) = tanh(z)
$$
A tanh function is a hyperbolic tangent defined as:
$$
tanh = \frac{sinh(x)}{cosh(x)} = \frac{\frac{1 - e^{-2x}}{2e^{-x}}}{\frac{1 + e^{-2x}}{2e^{-x}}}
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/504px-Sinh_cosh_tanh.svg.png)

Between this and the sigmoid function, tanh is usually preferred as it is centered at (0,0).
### ReLU - Restricted Linear Unit neuron
Neuron with an activation function of:
$$
f(z) = max(0, z)
$$
## Softmax output layer
Function that normalizes a **K** dimensional vector of **z** arbitrary values, into a **K** dimensional vector, where all elements add up to 1. This allows for the output of classification to be interpreted as a vector of probabilities with which a certain class was picked. Usually used with ***cross-entropy loss function training***.