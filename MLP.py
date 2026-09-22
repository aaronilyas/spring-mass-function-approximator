from torch import Tensor, nn, tensor
import torch
from torch.optim import Adam


class MLP(nn.Module):
    model: nn.Sequential
    optim: Adam

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)

    def feed_forward(self, t: Tensor) -> Tensor:
        x = self.model(t)
        return x

    def derivative(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.autograd.grad(inputs=t, outputs=x, create_graph=True)[0]

    def second_derivative(self, t: Tensor, x: Tensor) -> Tensor:
        first_derivative = self.derivative(t, x)
        return torch.autograd.grad(
            outputs=first_derivative, inputs=t, create_graph=True
        )[0]

    def train_network(
        self,
        epochs: int,
        spring_constant: float,
        mass: float,
        damping_coefficient: float,
        inital_condition_position: torch.Tensor,
        inital_condition_velocity: torch.Tensor,
    ) -> None:
        t_0 = torch.tensor([0])
        for i in range(epochs):
            independent_variable = torch.tensor([i]).float()
            independent_variable.requires_grad_(True)

            x = self.feed_forward(independent_variable)

            inital_velocity = self.derivative(
                t_0, self.feed_forward(torch.tensor([t_0]))
            )

            dx_dt = self.derivative(
                independent_variable, self.feed_forward(independent_variable)
            )
            d2x_dt2 = self.derivative(independent_variable, dx_dt)

            residual_ode = (
                (spring_constant * x) + (dx_dt * damping_coefficient) + (d2x_dt2 * mass)
            )

            residual_initial_position = (
                self.feed_forward(torch.tensor([0])) - inital_condition_position
            )

            residual_velocity = inital_velocity - inital_condition_velocity

            loss = residual_ode**2 + residual_initial_position**2 + residual_velocity**2

            self.model.zero_grad()

            loss.backward()

            self.optim.step()
