if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")

    centers, _ = lloyd_algorithm(x_train, 10, 0.01)

    fig, axs = plt.subplots(nrows=1, ncols=10)
    fig.subplots_adjust(wspace=0.1)

    for center, ax in zip(centers, axs):
        ax.imshow(np.reshape(center, (28, 28)))
        ax.axis('off')

    plt.savefig('A3pictures.png')
    plt.show()


if __name__ == "__main__":
    main()
