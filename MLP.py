from typing import List

from torch import Tensor, nn, ones_like, tensor
import torch
from torch.optim import Adam


class MLP(nn.Module):
    model: nn.Sequential
    optim: Adam

    def __init__(self) -> None:
        """
        This method is run when an instance of this class is created, and creates the neural network as well as assigns it to the model variable defined above.

        @author Aaron Ilyas
        """

        # Initialising parent class
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )

        # Utilising Adam optimizer because it on average converges on local minimum in a few number epochs than SGD.
        # This increases the likelihood of maintaining decent performance even if the user inputs a fewer number epochs.
        self.optim = torch.optim.Adam(self.parameters(), lr=0.002)

    def feed_forward(self, t: Tensor) -> Tensor:
        """
        Method accepts a tensor of various values of t, and passes them into the neural network, we then return the output.

        @author Aaron Ilyas
        """
        x = self.model(t)
        return x

    def derivative(self, t: Tensor, x: Tensor) -> Tensor:
        """
        This method takes the derivative of our position function (x) with respect to time.
        This method works even if each tensor that is passed in has more than one element but the dimensions must be consistent across both tensors.

        @author Aaron Ilyas
        """

        return torch.autograd.grad(
            inputs=t, outputs=x, grad_outputs=torch.ones_like(x), create_graph=True
        )[0]

    def second_derivative(self, t: Tensor, x: Tensor) -> Tensor:
        """
        This method takes the second derivative of our position function (x) with respect to time.
        This method works even if each tensor that is passed in has more than one element but the dimensions must be consistent across both tensors.

        @author Aaron Ilyas
        """
        first_derivative = self.derivative(t, x)
        return torch.autograd.grad(
            outputs=first_derivative,
            inputs=t,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
        )[0]

    def train_network(
        self,
        epochs: int,
        spring_constant: float,
        mass: float,
        damping_coefficient: float,
        actual_inital_condition_position: torch.Tensor,
        actual_inital_condition_velocity: torch.Tensor,
    ) -> None:
        """
        This method trains the neural network by defining a loss function,
        which is computed using the mean values for the error present in the values computed by the neural network for each point in time passed into it.

        @author Aaron Ilyas
        """

        # Defining a variable at time zero which is used to compute the networks predicted initial position
        t_0 = torch.tensor([0]).float()
        t_0.requires_grad_(True)

        for i in range(epochs):
            # Defining tensor consisting of 100 elements which range from 0 to 10 and are evenly spaced out. We then reshape that tensor from a 1D tensor to a 2D tensor with 100 rows and 1 column.
            independent_variable = torch.linspace(0, 10, 100).reshape(-1, 1)
            independent_variable.requires_grad_(True)

            # The initial position predicted by our network
            predicted_initial_position = self.feed_forward(t_0).float()

            # Computes a tensor consisting of each predicted position for each point in time present our input tensor.
            x = self.feed_forward(independent_variable)

            dx_dt = self.derivative(independent_variable, x)
            d2x_dt2 = self.derivative(independent_variable, dx_dt)

            # We are defining our residual to be equal to our ODE, which in its self should be equal to zero.
            # This means that the further away the value is from zero the larger the error on our networks predicted value for the ODE portion of our system.
            residual_ode = (
                (spring_constant * x) + (dx_dt * damping_coefficient) + (d2x_dt2 * mass)
            )

            # This measures how far our networks predicted initial position is from our actual initial position.
            residual_initial_position = (
                predicted_initial_position - actual_inital_condition_position
            )

            inital_velocity = self.derivative(t_0, predicted_initial_position)

            # This measures how far our networks predicted initial velocity is from our actual initial position.
            residual_velocity = inital_velocity - actual_inital_condition_velocity

            # We are defining our loss function to be the sum of the average error for each of our residuals squared.
            loss = (
                torch.mean(residual_ode**2)
                + torch.mean(residual_initial_position**2)
                + torch.mean(residual_velocity**2)
            )

            self.optim.zero_grad()

            loss.backward()

            self.optim.step()

    def predict_position_over_interval_of_time(self, start_t: int, stop_t: int) -> dict:
        """
        This method returns the predicted position values for each point in time from our starting point in time to our ending point in time.
        After doing that it returns a dictionary where the key is the point in time, and the value is the position at that point in time.

        @author Aaron Ilyas
        """

        self.model.eval()
        self.positions_at_each_time = {}
        with torch.no_grad():
            for i in range(start_t, stop_t):
                self.positions_at_each_time[str(i)] = self.feed_forward(
                    torch.tensor([i]).float()
                )

        return self.positions_at_each_time
