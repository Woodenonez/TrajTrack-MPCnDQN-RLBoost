import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(path: str) -> None:
    f = np.load(path)

    plt.figure()
    plt.plot(f["timesteps"], np.mean(f["results"], 1))
    plt.xlabel("Total number of steps taken")
    plt.ylabel("Mean return over %d evaluation episode" % len(f["results"][0]))
    plt.title("Training results")
    plt.show()