from MLP import MLP
import torch
import matplotlib.pyplot as plt


def collect_info_from_user_to_train_model(model: MLP) -> None:
    """
    This method collects all the necessary information from the user and then passes that into our train network function.

    @author Aaron Ilyas
    """

    epochs = int(input("Enter the number of training epochs: "))
    spring_constant = float(input("Enter the spring constant: "))
    mass = float(input("Enter the mass: "))
    damping_coefficient = float(input("Enter the damping coefficient: "))
    initial_position = float(input("Enter the initial position: "))
    initial_velocity = float(input("Enter the initial velocity: "))

    model.train_network(
        epochs=epochs,
        spring_constant=spring_constant,
        mass=mass,
        damping_coefficient=damping_coefficient,
        actual_inital_condition_position=torch.tensor(initial_position).float(),
        actual_inital_condition_velocity=torch.tensor(initial_velocity).float(),
    )


def main():
    model = MLP()
    collect_info_from_user_to_train_model(model)

    starting_time = input("Enter: the time you wish the graph to start at ")
    ending_time = input("Enter: the time you wish the graph to end at ")
    position_at_various_time_values = model.predict_position_over_interval_of_time(
        int(starting_time), int(ending_time)
    )

    keys = list(position_at_various_time_values.keys())
    values = list(position_at_various_time_values.values())

    plt.scatter(keys, values)
    plt.show()


if __name__ == "__main__":
    main()
