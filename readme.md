# NanoNet — Neural Network Library in C++

NanoNet is a fully custom neural network library implemented in C++ without external machine learning frameworks. It includes a custom tensor structure, matrix operations, forward and backward propagation, trainable layers, activation functions, loss functions, and an SGD optimizer. The project demonstrates how neural networks work at a low level by implementing each component manually.

## Features

### Core Components
- Custom Tensor2D implementation supporting matrix multiplication, element-wise operations, row reductions, and function application.
- Fully connected Dense layers with weight and bias updates.
- Activation layers including ReLU and Sigmoid.
- Mean squared error (MSE) loss function.
- SGD optimizer for parameter updates.

### Example Model
A multilayer perceptron (2 → 4 → 1) trained on the XOR dataset using forward and backward propagation.

## Project Structure

```
nanonet.cpp        # Full implementation and example usage
README.md
```

All components are implemented in a single file for simplicity. The architecture allows splitting the project into multiple headers and source files for future expansions.

## Build Instructions

### Compile

Requires a C++17 compiler:

```
g++ -std=c++17 -O2 nanonet.cpp -o nanonet
```

### Run

```
./nanonet
```

The program trains the XOR model for 10,000 epochs and prints the loss every 1000 epochs, followed by final predictions.

## Training Output Example

```
Epoch 0, loss = 0.27891
Epoch 1000, loss = 0.11572
Epoch 2000, loss = 0.06953
Epoch 3000, loss = 0.04322
Epoch 4000, loss = 0.03011
Epoch 5000, loss = 0.02274
Epoch 9000, loss = 0.01521

Trained XOR predictions:
Outputs =
 0.0121
 0.9853
 0.9732
 0.0218
```

## Implementation Details

### Forward Pass

```
output = activation(X * W + b)
```

### Backward Pass

Gradient calculations implemented manually:

```
dW = X^T * dL/dout
db = sum(dL/dout over batch)
dX = dL/dout * W^T
```

### Optimization

SGD update step:

```
W -= lr * dW
b -= lr * db
```