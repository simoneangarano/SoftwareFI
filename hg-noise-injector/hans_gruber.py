"""Hans Gruber training noise injector
This file encapsulate the noise injector to be used
in the training process
"""
import random
import warnings

import torch

LINE, SQUARE, RANDOM, ALL = "LINE", "SQUARE", "RANDOM", "ALL"


class HansGruberNI(torch.nn.Module):
    def __init__(self, error_model: str = LINE):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = error_model
        self.noise_data = list()

    def set_noise_data(self, noise_data: list = None):
        # make a subset of the errors
        self.noise_data = [i for i in noise_data if i["geometry_format"] == self.error_model]

    def forward(self, input):
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = input.clone()
        # TODO: It must be generalized for tensors that have more than 2 dim
        warnings.warn("Need to fix the HansGruber noise injector to support more than 2d dimension before use")
        # assert len(input.shape) == 2, f"Generalize this method to n-arrays {input.shape}\n{input}"
        # rows, cols = input.shape
        relative_error = random.choice(self.noise_data)
        relative_error = random.uniform(float(relative_error["min_relative"]), float(relative_error["max_relative"]))
        if self.error_model == LINE:
            # relative_errors = torch.FloatTensor(1, rows).uniform_(0, 1)
            rand_row = random.randrange(0, input.shape[0])
            output[rand_row, :].mul_(relative_error)
        # elif self.error_model == ErrorModel.COL:
        #     # relative_errors = torch.FloatTensor(1, cols).uniform_(0, 1)
        #     rand_col = random.randrange(0, input.shape[1])
        #     output[:, rand_col].mul_(relative_error)
        else:
            raise NotImplementedError
        print(input[input != output])

        return output
