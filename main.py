from MLP import MLP
import torch
import matplotlib.pyplot as plt


def main():
    model = MLP()
    model.train_network(20, 2, 4, 16, torch.tensor(2), torch.tensor(2))
    positions = model.predict_position_over_interval_of_time(0, 10)
    times = list(positions.keys())
    positions = list(positions.values())
    plt.scatter(times, positions)
    plt.show()


if __name__ == "__main__":
    main()
