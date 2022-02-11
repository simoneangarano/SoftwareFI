"""Hans Gruber training noise injector
This file encapsulates the noise injector to be used
in the training process
"""
import random
import torch

LINE, SQUARE, RANDOM, ALL = "LINE", "SQUARE", "RANDOM", "ALL"


class HansGruberNI(torch.nn.Module):
    def __init__(self, error_model: str = LINE, p: float = 0.3, inject_epoch: int = 0):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = error_model
        self.noise_data = list()
        self.p = p  # fraction of the samples which the injection is applied to
        self.inject_epoch = inject_epoch  # how many epochs before starting the injection

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
        # Power Law parameters for the Functional Units
        power_law_fus = [
            (1.0728769e-07, 1.0868737), (2.0230031, 1.0568325), (8.1847715e-08, 1.082071), (136027.72, 27.1194),
            (3.0, 1.0678725), (0.03517608, 1.189603), (3.4028237e+38, 443107.0), (2.0, 1.4543958),
            (0.010238367, 1.1181921), (1.396856e-09, 1.0846596), (2.6865074e-10, 1.0769672), (1.3970158e-09, 1.085144),
            (0.66699225, 23.798765), (0.66699225, 23.798765), (0.66699225, 23.922783), (0.75000001, 121435080.0),
            (0.61141304, 3.4316596), (0.75000001, 121435080.0), (0.0, 1.08212), (7.0958774e-08, 1.082116),
            (0.0, 1.08212)
        ]

        alpha, x_min = random.choice(power_law_fus)
        r = random.random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        # print(relative_error)
        return relative_error
        #return 27.119592052269397

    def inject(self, forward_input: torch.Tensor, p: float) -> torch.Tensor:
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = forward_input.clone()
        b, c, h, w = output.shape
        # select the samples which the injection is applied to with probability p
        sampled_indexes = torch.bernoulli(torch.ones(b) * p)
        sampled_indexes = sampled_indexes > 0

        if self.error_model == LINE:
            # select the row
            rand_row = torch.randint(h, size=(1,))
            if torch.bernoulli(torch.ones(1) * 0.5):
                output[sampled_indexes, :, :, rand_row] = output[sampled_indexes, :, :, rand_row].mul_(
                    self.random_relative_error)
            else:
                output[sampled_indexes, :, rand_row, :] = output[sampled_indexes, :, rand_row, :].mul_(
                    self.random_relative_error)

        elif self.error_model == SQUARE:
            raise NotImplementedError("Implement SQUARE error model first")
        elif self.error_model == RANDOM:
            raise NotImplementedError("Implement RANDOM first")
        elif self.error_model == ALL:
            raise NotImplementedError("Implement ALL first")

        return output

    def forward(self, forward_input: torch.Tensor, inject: bool = True, current_epoch: int = 0) -> torch.Tensor:
        r"""Perform a 'forward' operation to simulate the error model injection
        in the training process
        :param inject: whether to apply injection or not at test time
        :param forward_input: torch.Tensor input for the forward
        :return: processed torch.Tensor
        """
        output = forward_input
        if self.training:
            if current_epoch >= self.inject_epoch:
                # inject noise to each sample with probability p
                output = self.inject(forward_input, self.p)
        else:
            if inject:
                # inject noise to all samples
                output = self.inject(forward_input, 1)

        return output
