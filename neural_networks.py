import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = X @ self.W1 + self.b1
        if self.activation_fn == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation_fn == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        else:
            raise ValueError("Unsupported activation function")
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.z2
        out = self.a2
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        delta2 = self.a2 - y
        dW2 = self.a1.T @ delta2 / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        if self.activation_fn == 'tanh':
            da1 = delta2 @ self.W2.T
            dz1 = da1 * (1 - np.tanh(self.z1) ** 2)
        elif self.activation_fn == 'relu':
            da1 = delta2 @ self.W2.T
            dz1 = da1 * (self.z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid_z1 = 1 / (1 + np.exp(-self.z1))
            da1 = delta2 @ self.W2.T
            dz1 = da1 * sigmoid_z1 * (1 - sigmoid_z1)
        else:
            raise ValueError("Unsupported activation function")
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # TODO: update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # TODO: store gradients for visualization
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

        pass

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # TODO: Hyperplane visualization in the hidden space
    W2 = mlp.W2.flatten()
    b2 = mlp.b2.flatten()
    hs_min = hidden_features.min(axis=0) - 1
    hs_max = hidden_features.max(axis=0) + 1
    h1, h2 = np.meshgrid(np.linspace(hs_min[0], hs_max[0], 30),
                         np.linspace(hs_min[1], hs_max[1], 30))
    if W2[2] != 0:
        h3 = (-W2[0]*h1 - W2[1]*h2 - b2)/W2[2]
        ax_hidden.plot_surface(h1, h2, h3, alpha=0.3)

    ax_hidden.set_xlim(hs_min[0], hs_max[0])
    ax_hidden.set_ylim(hs_min[1], hs_max[1])
    ax_hidden.set_zlim(-1.5, 1.5)  # Fixed height axis

    # TODO: Distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    grid = np.c_[xx.ravel(), yy.ravel()]
    mlp.forward(grid)
    hidden_grid_features = mlp.a1
    ax_hidden.scatter(hidden_grid_features[:, 0], hidden_grid_features[:, 1], hidden_grid_features[:, 2], c='grey', alpha=0.1)

    # TODO: Plot input layer decision boundary
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z > 0, alpha=0.3, levels=[-1, 0, 1], colors=['blue', 'red'])
    ax_input.contour(xx, yy, Z, levels=[0], colors='black')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    input_nodes = [(0, i) for i in range(mlp.W1.shape[0])]
    hidden_nodes = [(1, i) for i in range(mlp.W1.shape[1])]
    output_node = [(2, 0)]
    for node in input_nodes:
        ax_gradient.add_patch(Circle(node, radius=0.1, color='blue'))
    for node in hidden_nodes:
        ax_gradient.add_patch(Circle(node, radius=0.1, color='green'))
    for node in output_node:
        ax_gradient.add_patch(Circle(node, radius=0.1, color='red'))
    scale_factor = 10
    for i, input_node in enumerate(input_nodes):
        for j, hidden_node in enumerate(hidden_nodes):
            grad = mlp.dW1[i, j]
            linewidth = np.abs(grad) * scale_factor
            ax_gradient.plot([input_node[0], hidden_node[0]], [input_node[1], hidden_node[1]], color='black', linewidth=linewidth)
    for i, hidden_node in enumerate(hidden_nodes):
        grad = mlp.dW2[i, 0]
        linewidth = np.abs(grad) * scale_factor
        ax_gradient.plot([hidden_node[0], output_node[0][0]], [hidden_node[1], output_node[0][1]], color='black', linewidth=linewidth)

    ax_gradient.axis('on')

    for node, label in zip(input_nodes, ['x1', 'x2']):
        ax_gradient.text(node[0], node[1]+0.2, label, ha='center')

    for node, label in zip(hidden_nodes, ['h1', 'h2', 'h3']):
        ax_gradient.text(node[0], node[1]+0.2, label, ha='center')

    for node, label in zip(output_node, ['y']):
        ax_gradient.text(node[0], node[1]+0.2, label, ha='center')

    step = frame * 10
    ax_hidden.set_title(f'Hidden Space at Step {step}')
    ax_input.set_title(f'Input Space at Step {step}')
    ax_gradient.set_title(f'Gradients at Step {step}')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)