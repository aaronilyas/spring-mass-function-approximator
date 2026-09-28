import sympy
from sympy import Basic, Eq, dsolve, exp,symbols, Function
from typing import cast
def compute_solution_to_second_order_ode_from_spring_mass_system(spring_constant:float, damping_coefficient:float, mass:float,initial_position: float, initial_velocity: float):
    """
    This class defines the symbolic representation of the second order differential equation that governs the spring mass system, and returns a symbolic representation of it.

    @author Aaron Ilyas
    """
    x = Function("x")
    t = symbols("t")
    position =  spring_constant*x(t)
    velocity = damping_coefficient*x(t).diff(t)
    acceleration = mass*(x(t).diff()).diff(t)
    ode = Eq(acceleration + velocity + position, 0)
    solution = dsolve(ode,ics={x(t).subs(t,0): initial_position, x(t).diff(t).subs(t,0): initial_velocity})
    return cast(Eq,solution)
