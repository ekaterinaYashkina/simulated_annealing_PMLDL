# second version
import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nn import Classifier
from sim_anneal import SimulatedAnnealing
import torch.nn as nn
import matplotlib.pyplot as plt
import time


import sys
import argparse

parser1 = argparse.ArgumentParser(allow_abbrev=False)
parser1.add_argument("--start_T", type = float, default=11, help = "starting temperature for SA")
parser1.add_argument("--iters", type = int, default=20000, help = "number of iterations")
parser1.add_argument("--ann_schedule", type = bool, default=True, help = "whether to use scheduling for T decrease")
parser1.add_argument("--ann_iters", type = int, default=50, help = "period for annealing scheduling")
parser1.add_argument("--cool_rate", type = float, default=0.9, help = "proportion of T to keep after annealing")
parser1.add_argument("--mus", type = str, default='weights', help = "how to generate new model weights' mus:"
                                                                    "'weights' - take mu as the current weight"
                                                                    "'fixed' - generate mus for all weights beforehand")
parser1.add_argument("--std", type = int, default=2, help = "std for generating new weights from normal distribution")
parser1.add_argument("--plot", type = bool, default=True, help = "whether to plot the loss and accuracy change")
parser1.add_argument("--optimizer", type = str, default='sa', help = "sa or sgd optimizer")

arg = parser1.parse_args()

# either train on cuda or cpu, in torch we need to assign the device for all instances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""

Train the model using simulated annealing as optimizer

X - matrix of features
Y - labels to predict
model - the model for prediction
criterion - loss function to optimize
min_T - minimum temperature to stop the algorithm
epochs - number of iterations to perform to stop the algorithm

"""
def train_sa(X, Y, model, optimizer, criterion, min_T = 1e-8, epochs = 20000):

    history = []
    ite = 0
    while ite < epochs and optimizer.T > min_T:

        # Find the current loss and update weights using simulated annealing
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, Y)
            optimizer.step()

        # Print the information every 5 iterations
        if ite % 5 == 0:
            _, predicted = torch.max(outputs.data, 1)
            accuracy = torch.sum(Y == predicted).item() / len(x_train)
            print('Epoch [%d/%d]  Loss: %.4f Accuracy: %.4f SA_temp %.6f'
                  % (ite + 1, epochs, loss.item(), accuracy, optimizer.T))

        history.append((loss, accuracy))
        ite += 1

    return history, model


def train_sgd(X, Y, model, optimizer, criterion, epochs = 20000):

    history = []
    ite = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # Print the information every 5 iterations
        if ite % 5 == 0:
            _, predicted = torch.max(outputs.data, 1)
            accuracy = torch.sum(Y == predicted).item() / len(x_train)
            print('Epoch [%d/%d]  Loss: %.4f Accuracy: %.4f '
                  % (ite + 1, epochs, loss.item(), accuracy))

        history.append((loss, accuracy))
        ite += 1

    return history, model


"""

Calculate the performance on the test set

x_test - matrix of features
y_test - labels to predict
model - the model for prediction
criterion - loss function to optimize

"""
def test(x_test, y_test, model, criterion):
    X_t = Variable(torch.Tensor(x_test).float())
    Y_t = Variable(torch.Tensor(y_test).long())
    outputs_t = model(X_t.to(device))
    loss = criterion(outputs_t, Y_t.to(device))

    _, predicted_t = torch.max(outputs_t.data, 1)
    acc = torch.sum(Y_t.cpu() == predicted_t.cpu()).item() / len(x_test)

    print("Test set loss - {}, accuracy - {}".format(loss, acc))
    return loss, acc



"""

Plot the history of the model performance

history - list of tuples [(loss, acc), ...]

"""
def plot_change(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Loss and accuracy with SA optimization')

    y = [history[i][0] for i in range(len(history))]
    x = [i for i in range(1, len(history)+1)]

    ax1.plot(x, y)

    y1 = [history[i][1] for i in range(len(history))]
    x1 = [i for i in range(1, len(history)+1)]

    ax2.plot(x1, y1)

    ax1.set(xlabel='Iterations', ylabel='Loss')
    ax2.set(xlabel='Iterations', ylabel='Accuracy')


    plt.show()


# iris dataset from sklearn


iris = load_iris()
x_data=iris.data
y_data=iris.target


# split on train and test sets
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)



# The energy function - cross-entropy loss
criterion = torch.nn.CrossEntropyLoss()

model = Classifier()
model = model.to(device)


if arg.optimizer == 'sa':
    optimizer = SimulatedAnnealing(params = model.parameters(), device = device, model = model, features = torch.Tensor(x_train).float(),
                               labels = torch.Tensor(y_train).long(), loss = nn.CrossEntropyLoss(),
                               T_init = arg.start_T, cool_rate = arg.cool_rate,
                               annealing_schedule = arg.ann_schedule, ann_iter = arg.ann_iters, mus = arg.mus, std = arg.std)
elif arg.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

else:
    raise ValueError("Optimizer should be sa or sgd")


ite = 0
min_T = 1e-8
epochs = arg.iters


X = Variable(torch.Tensor(x_train).float()).to(device)
Y = Variable(torch.Tensor(y_train).long()).to(device)


if arg.optimizer == "sa":
    time1 = time.time()
    history, model = train_sa(X, Y, model, optimizer, criterion, min_T=min_T, epochs=epochs)
    test(x_test, y_test, model, criterion)
    time2 = time.time() - time1
else:
    time1 = time.time()
    history, model = train_sgd(X, Y, model, optimizer, criterion,  epochs=epochs)
    test(x_test, y_test, model, criterion)
    time2 = time.time() - time1

print("Time for execution - {}".format(time2))

if arg.plot:
    plot_change(history)



