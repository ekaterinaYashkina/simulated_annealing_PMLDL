import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import numpy as np


"""

Pytorch optimization class that is able to update model weights
The used algorithm - simulated annealing

The performance is calculated based on cross entropy loss.
The weights are updated using 2 modes: weights - generate a new value form normal distribution
with mu = current_weight or fixed - generate the mus for all the weights beforehand and use them
in every generation.

This value can be provided in initialization

"""

class SimulatedAnnealing(Optimizer):

    """

    params - the model parameters
    device - cuda or cpu, type of accelerator used on your machine
    T_init - which temperature to use as the start one

    cool_rate - which proportion of previous temp to use in another iteration
    annealing_schedule - True/False - whether to use some pereodicity to change the temperature
    ann_iter - if annealing_schedule is True, then must be provided; period of scheduling
    model - model to optimize
    features - X
    labels - Y, to predict
    loss - loss function (energy) in optimization, by default will be - cross entropy loss
    mus - 'weights' - generate a new value form normal distribution
with mu = current_weight, 'fixed' - generate the mus for all the weights beforehand and use them
in every generation
    std - std to use in weights generation using normal distribution


    """

    def __init__(self, params, device, T_init = 11, cool_rate = 0.9, annealing_schedule = True, ann_iter = 30,
                 model = None, features = None, labels = None, loss = None, mus = 'weights', std = 2):

        super(SimulatedAnnealing, self).__init__(params, defaults = {})
        self.device = device
        self.T = T_init
        self.cool_rate = cool_rate
        self.annealing_schedule = annealing_schedule

        if self.annealing_schedule:
            assert ann_iter is not None

        self.ann_iter = ann_iter


        self.model = model
        self.features = features
        self.labels = labels

        if loss is None:
            self.loss = nn.CrossEntropyLoss
        else:
            self.loss = loss
        self.ite = 0


        if mus != 'weights' and mus !="fixed":
            raise ValueError("Wrong mus for weights generation provided. Use default or 'weights' or 'fixed' ")

        else:
            self.mus_mode = mus
            self.std = std
            self.mus = {}


        if mus == 'fixed':
            for name, param in self.model.state_dict().items():
                self.mus[name] = np.random.choice(np.arange(0, 3))




    """
    
    
    Update the weights of the model using SA algorithm.
    
    1. Generate the new values, calculate the performance of the model with the provided loss function.
    2. Update the values if the new loss is better than the previous (new < current)
    3. If the new loss is worse, then if U(0, 1) < acceptance_ration (new, cur) also update the values, otherwise,
    keep the same weights.
    4. Change the temperature using some schedule (or every iteration)
    
    
    """


    def step(self, closure = None):

        cur_weights = self.model(self.features.to(self.device))
        cur_loss = self.loss(cur_weights, self.labels.type(torch.LongTensor).to(self.device))
        old_state_dict = {}
        for key in self.model.state_dict():
            old_state_dict[key] = self.model.state_dict()[key].clone()

        for name, param in self.model.state_dict().items():
            if (len(param.shape) == 2):
                if self.mus_mode == "fixed":
                    new_w = torch.Tensor(np.random.normal(loc=self.mus[name], scale=self.std, size=(param.shape[0], param.shape[1])))
                else:
                    new_w = torch.Tensor(np.random.normal(loc = old_state_dict[name].cpu(), scale = self.std, size = (param.shape[0], param.shape[1])))

            else:
                if self.mus_mode == "fixed":
                    new_w = torch.Tensor(np.random.normal(loc=self.mus[name], scale=self.std, size=param.shape[0]))
                else:
                    new_w = torch.Tensor(np.random.normal(loc = old_state_dict[name].cpu(), scale = self.std, size = param.shape[0]))


            new_w = new_w.to(self.device)
            self.model.state_dict()[name].copy_(new_w)

        Y_pred = self.model(self.features.to(self.device))

        new_loss = self.loss(Y_pred, self.labels.type(torch.LongTensor).to(self.device))

        if (new_loss >= cur_loss):
            if self.device == 'cuda':
              jumpProb = np.exp(-(new_loss.cpu() - cur_loss.cpu()) / self.T)
            else:
              jumpProb = np.exp(-(new_loss - cur_loss) / self.T)
            if (np.random.uniform(0, 1) > jumpProb):
                self.model.load_state_dict(old_state_dict)

        if self.annealing_schedule:
            if self.ite % self.ann_iter == 0:
              self.T = self.cool_rate*self.T
        else:
            self.T = self.cool_rate * self.T

        self.ite+=1