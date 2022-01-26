"""Hans Gruber training noise injector
This file encapsulates the noise injector to be used
in the training process
"""
import random
import warnings
import torch

LINE, SQUARE, RANDOM, ALL = "LINE", "SQUARE", "RANDOM", "ALL"


class HansGruberNI(torch.nn.Module):
    def __init__(self, error_model: str = LINE, p: float = 0.3):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = error_model
        self.noise_data = list()
        self.p = p  # fraction of the samples which the injection is applied to

    def set_noise_data(self, noise_data: list = None) -> None:
        r"""Set the noise data that we extract and parse from radiation experiments
        The noise data is extracted form a CSV file, pass only a numpy array to the function
        """
        # make a subset of the errors
        self.noise_data = [i for i in noise_data if i["geometry_format"] == self.error_model]

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

    def inject(self, forward_input: torch.Tensor, p: float) -> torch.Tensor:
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = forward_input.clone()
        B, C, H, W = output.shape
        # select the samples which the injecttion is applied to with probability p
        sampled_indexes = torch.bernoulli(torch.ones(B) * p)
        sampled_indexes = sampled_indexes > 0

        if self.error_model == LINE:
            # select the row
            rand_row = torch.randint(H, size=(1,))
            if torch.bernoulli(torch.ones(1) * 0.5):
                output[sampled_indexes, :, :, rand_row] = output[sampled_indexes, :, :, rand_row].mul_(
                    self.random_relative_error())
            else:
                output[sampled_indexes, :, rand_row, :] = output[sampled_indexes, :, rand_row, :].mul_(
                    self.random_relative_error())

        elif self.error_model == SQUARE:
            raise NotImplementedError("Implement SQUARE error model first")
        elif self.error_model == RANDOM:
            raise NotImplementedError("Implement RANDOM first")
        elif self.error_model == ALL:
            raise NotImplementedError("Implement ALL first")
        #print(forward_input[forward_input != output])

        return output

    def forward(self, forward_input: torch.Tensor) -> torch.Tensor:
        r"""Perform a 'forward' operation to simulate the error model injection
        in the training process
        :param forward_input: torch.Tensor input for the forward
        :return: processed torch.Tensor
        """
        if self.training:
            # inject noise to each sample with probability p
            output = self.inject(forward_input, self.p)
        else:
            # inject noise to all samples
            output = self.inject(forward_input, 1)

        return output
