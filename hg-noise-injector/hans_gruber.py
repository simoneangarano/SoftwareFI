"""Hans Gruber training noise injector
This file encapsulate the noise injector to be used
in the training process
"""
import random

import torch


class HansGruberNI(torch.nn.Module):
    def __init__(self, input_features, output_features, error_model=None):
        super(HansGruberNI, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = torch.nn.Parameter(torch.empty(output_features, input_features))

        # Error model necessary for the forward
        self.error_model = error_model

        # Not a very smart way to initialize weights
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, input):
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = input.clone()
        assert len(input.shape) == 2, "Generalize this method to n-arrays"
        rows, cols = input.shape
        if self.error_model == "ROW":
            relative_errors = torch.FloatTensor(1, rows).uniform_(0, 1)
            rand_row = random.randrange(0, rows)
            output[rand_row, :].mul_(relative_errors)
        elif self.error_model == "COL":
            relative_errors = torch.FloatTensor(1, cols).uniform_(0, 1)
            rand_col = random.randrange(0, cols)
            output[:, rand_col].mul_(relative_errors)
        elif self.error_model == "BLOCK":
            raise NotImplementedError
        elif self.error_model == "ALL":
            raise NotImplementedError
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f'input_features={self.input_features}, output_features={self.output_features}'
