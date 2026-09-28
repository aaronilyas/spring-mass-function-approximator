from sympy import plot, symbols
import numpy as np
from MLP import MLP
import torch
import matplotlib.pyplot as plt

import compute_algebraic_solution


def collect_info_from_user() -> tuple:
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

    return epochs,spring_constant, mass, damping_coefficient, initial_position, initial_velocity

def main():
    model = MLP()
    epochs, spring_constant, mass, damping_coefficient, initial_position, initial_velocity = collect_info_from_user()
    model.train_network(
        epochs=epochs,
        spring_constant=spring_constant,
        mass=mass,
        damping_coefficient=damping_coefficient,
        actual_initial_condition_position=torch.tensor(initial_position).float(),
        actual_initial_condition_velocity=torch.tensor(initial_velocity).float(),
    )

    starting_time = input("Enter: the time you wish the graph to start at ")
    ending_time = input("Enter: the time you wish the graph to end at ")
    position_at_various_time_values = model.predict_position_over_interval_of_time(
        int(starting_time), int(ending_time)
    )
    
    symbolic_solution = compute_algebraic_solution.compute_solution_to_second_order_ode_from_spring_mass_system(spring_constant,damping_coefficient,mass,initial_position,initial_velocity)

    keys = list(position_at_various_time_values.keys())
    values = list(position_at_various_time_values.values())

    plt.scatter(keys, values)
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()
   
    list_of_values_at_times = []
    list_of_times = []
    t = symbols('t')
    for i in range(int(starting_time),int(ending_time)):
        position_at_time = symbolic_solution.rhs.subs(t,i)
        list_of_values_at_times.append(position_at_time)
        list_of_times.append(i)

    list_of_values_at_times = np.array(list_of_values_at_times)
    list_of_times = np.array(list_of_times)
     
    plt.scatter(list_of_times,list_of_values_at_times)
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()



if __name__ == "__main__":
    main()
