"""Hans Gruber training noise injector
This file encapsulates the noise injector to be used
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

    def set_noise_data(self, noise_data: list = None) -> None:
        r"""Set the noise data that we extract and parse from radiation experiments
        The noise data is extracted form a CSV file, pass only a numpy array to the function
        """
        # make a subset of the errors
        self.noise_data = [i for i in noise_data if i["geometry_format"] == self.error_model]

    @property
    def random_relative_error(self) -> float:
        r"""Generator for relative errors to be injected on the training
        We have seen in the past relative error distributions that follow a Power Law PDF
        so we will use the approach proposed at https://arxiv.org/abs/1208.3524
        We will implement the function based on https://stats.stackexchange.com/a/406705
        Example:
        x_min, alpha, r = 5, 2.5, random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        :return: the calculated relative_error
        """
        # TODO: Generalize the random generation to the values observed on GEMM output
        x_min, alpha, r = 1.0728769e-07, 1.0868737e+00, random.random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        return relative_error

    def forward(self, forward_input: torch.Tensor) -> torch.Tensor:
        r"""Perform a 'forward' operation to simulate the error model injection
        in the training process
        :param forward_input: torch.Tensor input for the forward
        :return: processed torch.Tensor
        """
        # TODO: How to inject the error model? Is it static for the whole training?
        #  I believe we should randomize it, let's say: we pick a given
        #  layer and at each forward we randomly sample a certain feature
        #  map to corrupt among all the features

        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = forward_input.clone()
        # TODO: It must be generalized for tensors that have more than 2 dim
        warnings.warn("Need to fix the HansGruber noise injector to support more than 2d dimension before use")
        # TODO: put 1 relative error that is critical for the chosen network
        #  line or column

        if self.error_model == LINE:
            # relative_errors = torch.FloatTensor(1, rows).uniform_(0, 1)
            rand_row = random.randrange(0, forward_input.shape[0])
            output[rand_row, :].mul_(self.random_relative_error)
        elif self.error_model == SQUARE:
            raise NotImplementedError("Implement SQUARE error model first")
        elif self.error_model == RANDOM:
            raise NotImplementedError("Implement RANDOM first")
        elif self.error_model == ALL:
            raise NotImplementedError("Implement ALL first")
        print(forward_input[forward_input != output])

        return output
