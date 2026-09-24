# Spring Mass Function Approximator

This project uses a neural network to approximate the position of a mass attached to a spring over a given interval of time.

Instead of training the neural network using a dataset containing known position values, we train the network by defining a loss function based on the differential equation which describes a damped spring-mass system. This means that we can train the network by measuring how well its predicted position function satisfies the differential equation as well as the initial conditions provided by the user.

The differential equation we are attempting to satisfy is:

$$
m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0
$$

Where:

- `m` represents the mass.
- `c` represents the damping coefficient.
- `k` represents the spring constant.
- `x` represents the position of the mass.
- `t` represents time.

## How It Works

The neural network accepts a point in time as its input and returns the predicted position of the mass at that point in time.

The network currently consists of:

- One input neuron representing time.
- Two hidden layers containing 24 neurons each.
- `Tanh` activation functions between each hidden layer.
- One output neuron representing the predicted position.

PyTorch's automatic differentiation is then used to take the first and second derivatives of our networks predicted position with respect to time.

This allows us to calculate:

$$
\frac{dx}{dt}
$$

and:

$$
\frac{d^2x}{dt^2}
$$

which can then be placed into our differential equation.

We are defining the residual of our ODE as:

$$
kx + c\frac{dx}{dt} + m\frac{d^2x}{dt^2}
$$

The value of this equation should ideally be equal to zero. This means that the further away the value is from zero, the larger the error present in our networks predicted position function.

## Loss Function

The loss function is calculated using three different residuals.

The first residual measures how closely our predicted position function satisfies the differential equation.

The second residual measures how far our networks predicted initial position is from the initial position entered by the user.

The third residual measures how far our networks predicted initial velocity is from the initial velocity entered by the user.

We then square each of these residuals, calculate their average values and add them together to compute our final loss.

The network uses the Adam optimizer with a learning rate of `0.002`. Adam is being used because it generally converges on a local minimum in fewer epochs than standard stochastic gradient descent, increasing the likelihood that the network can still produce a reasonable approximation when using a smaller number of training epochs.

## Training

During each training epoch we create a tensor consisting of 100 evenly spaced points in time ranging from `0` to `10`.

The neural network predicts the position for each of these points. We then use automatic differentiation to calculate the predicted velocity and acceleration before computing the residuals used by our loss function.

The user can provide the following values before training begins:

- Number of training epochs.
- Spring constant.
- Mass.
- Damping coefficient.
- Initial position.
- Initial velocity.

## Predicting Position

After the network has been trained, the user can provide a starting and ending point in time.

The model predicts the position of the mass at each integer point in time within this interval and stores those values inside a dictionary, where each key represents a point in time and its value represents the predicted position at that point in time.

These values are then displayed using a scatter plot with Matplotlib.

## Project Structure

### `MLP.py`

This file contains the `MLP` class which is responsible for creating and training our neural network.

It also contains the methods used to calculate the first and second derivatives of our predicted position function as well as the method used to predict positions after training has finished.

### `main.py`

This file is responsible for collecting the required values from the user, training the neural network and displaying the resulting predicted positions using Matplotlib.

### `requirements.txt`

This file contains the Python packages and versions used by the project.

## Installation

Clone the repository:

```bash
git clone https://github.com/aaronilyas/spring-mass-function-approximator.git
```

Move into the project directory:

```bash
cd spring-mass-function-approximator
```

It is recommended to create a virtual environment before installing the required packages.

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running The Program

The program can be started using:

```bash
python main.py
```

The program will then ask for the values required to train the model.

For example:

```text
Enter the number of training epochs:
Enter the spring constant:
Enter the mass:
Enter the damping coefficient:
Enter the initial position:
Enter the initial velocity:
```

After training has finished, the program will ask for the starting and ending times that should be used when generating the graph.

A scatter plot will then be displayed containing the position predicted by the neural network for each point in time.

## Technologies Used

- Python
- PyTorch
- Matplotlib
- Automatic Differentiation
- Multilayer Perceptron Neural Network

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Author

Aaron Ilyas
