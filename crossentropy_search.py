if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    return_dict = {}

    model_network_1 = nn.Sequential(LinearLayer(2, 2), SoftmaxLayer())
    model_network_2 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), SoftmaxLayer())
    model_network_3 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SoftmaxLayer())
    model_network_4 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SoftmaxLayer())
    model_network_5 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), SoftmaxLayer())

    nn_models = [model_network_1, model_network_2, model_network_3, model_network_4, model_network_5]
    nn_names = ['LinReg', 'OneLayerSigmoid', 'OneLayerReLU', 'TwoLayersSigRe', 'TwoLayersReSig']

    epochs = 2000
    batch = 2 ** 5
    learning_rate = 10 ** -3

    for i, model in enumerate(nn_models):
        model_dict = {"train": [], "val": [], "model": []}
        train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch, shuffle=True)
        criterion = CrossEntropyLossLayer()
        optimizer = SGDOptimizer(model.parameters(), learning_rate)

        train_dict = train(train_loader, model, criterion, optimizer, val_loader, epochs)

        model_dict["train"] = train_dict["train"]
        model_dict["val"] = train_dict["val"]
        model_dict["model"] = model

        return_dict[nn_names[i]] = model_dict

    return return_dict



@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for (x,y) in dataloader:
            y_hat = model(x)
            class_predict = torch.argmax(y_hat, dim=1)

            for j in range(len(class_predict)):
                total_count += 1

                if y[j] == class_predict[j]:
                    correct_count += 1

    return correct_count/total_count


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    batches = 2 ** 5
    dataloader_test = DataLoader(dataset_test, batch_size=batches, shuffle=True)

    for nn_name, data in ce_configs.items():
        score = accuracy_score(data['model'], dataloader_test)
        print(f'{nn_name} accuracy score:', score)
        plot_model_guesses(dataloader_test, data['model'], title=f'Cross Entropy Search {nn_name}')

    for nn_name, data in ce_configs.items():
        train_values = data['train']
        val_values = data['val']

        plt.plot(train_values, label=f'{nn_name} - Train')
        plt.plot(val_values, label=f'{nn_name} - Val')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Val Loss')
    plt.legend(fontsize='small')
    plt.ylim(0,0.8)
    #plt.savefig('Crossentropyplot.pdf')
    plt.show()

if __name__ == "__main__":
    main()
