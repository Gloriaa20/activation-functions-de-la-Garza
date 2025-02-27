Activation Functions in Neural Networks

This project implements various activation functions used in neural networks and generates graphs to visualize their behavior.


📌 Implemented Functions

Sigmoid: f(x) = 1 / (1 + e^(-x))

Tanh: f(x) = tanh(x)

ReLU (Rectified Linear Unit): f(x) = max(0, x)

Leaky ReLU: f(x) = x if x > 0, alpha * x if x <= 0

Softmax: f(x) = e^x / sum(e^x)

ELU (Exponential Linear Unit): f(x) = x if x > 0, alpha * (e^x - 1) if x <= 0

Swish: f(x) = x * sigmoid(beta * x)


🚀 Requirements

Make sure you have the following dependencies installed before running the code:
pip install numpy matplotlib


🖥️ Usage

Save the code in a file named activations.py and run it with:
python activations.py


📊 Visualization

The code will generate graphs for each activation function, showing their behavior over a range of values.


🛠️ Author and Contribution

Created to demonstrate activation functions in neural networks. Contributions and improvements are welcome! 🚀
