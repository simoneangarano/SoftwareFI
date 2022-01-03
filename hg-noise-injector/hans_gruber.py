"""Hans Gruber training noise injector
This file encapsulate the noise injector to be used
in the training process
"""
import enum
import random
import warnings

import torch


class ErrorModel(enum.Enum):
    ROW, COL, BLOCK, ALL = range(4)


class HansGruberNI(torch.nn.Module):
    def __init__(self, error_model: ErrorModel = ErrorModel.ROW):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = error_model

    def forward(self, input):
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = input.clone()
        # TODO: Need to fix it before use
        warnings.warn("Need to fix the HansGruber noise injector to support more than 2d dimention before use")
        # assert len(input.shape) == 2, f"Generalize this method to n-arrays {input.shape}\n{input}"
        # rows, cols = input.shape
        relative_error = random.uniform(0, 100)

        if self.error_model == ErrorModel.ROW:
            # relative_errors = torch.FloatTensor(1, rows).uniform_(0, 1)
            rand_row = random.randrange(0, input.shape[0])
            output[rand_row, :].mul_(relative_error)
        elif self.error_model == ErrorModel.COL:
            # relative_errors = torch.FloatTensor(1, cols).uniform_(0, 1)
            rand_col = random.randrange(0, input.shape[1])
            output[:, rand_col].mul_(relative_error)
        elif self.error_model == ErrorModel.BLOCK:
            raise NotImplementedError
        elif self.error_model == ErrorModel.ALL:
            raise NotImplementedError
        return output
