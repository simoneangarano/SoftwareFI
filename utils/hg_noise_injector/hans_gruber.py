"""Hans Gruber training noise injector
This file encapsulates the noise injector to be used
in the training process
"""

import random
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn


def generate_4Dmask(
    shape: Tuple[int, int, int, int],
    batch_ids: List[int] = [],
    channel_ids: List[int] = [],
    row_ids: List[int] = [],
    column_ids: List[int] = [],
) -> torch.Tensor:
    b, c, w, h = shape
    batch_ids = [idx for idx in range(b) if idx in batch_ids]
    mask = torch.zeros((b, c, w, h))
    mask_2d = torch.zeros((w, h))

    r, col = len(row_ids), len(column_ids)
    if r and col:
        # single or square
        for row_id in row_ids:
            for column_id in column_ids:
                mask_2d[row_id, column_id] = 1

    elif r:
        # line - rows
        for row_id in row_ids:
            mask_2d[row_id, :] = 1

    elif col:
        # line - columns
        for column_id in column_ids:
            mask_2d[:, column_id] = 1

    else:
        # all
        mask_2d = torch.ones((w, h))

    def set_mask(batch_id, channel_id):
        mask[batch_id, channel_id] = mask_2d

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(set_mask, batch_id, channel_id)
            for batch_id in batch_ids
            for channel_id in channel_ids
        ]
        for future in futures:
            future.result()  # Wait for all threads to complete

    return mask > 0


def generate_2Dmask(shape, batch_ids=[]):
    b, c = shape
    batch_ids = [idx for idx in range(b) if batch_ids[idx]]

    rand_c = torch.bernoulli(torch.ones(c) * 0.3) > 0
    channel_ids = [idx for idx in range(c) if rand_c[idx]]

    mask = torch.zeros((b, c))

    for batch_id in batch_ids:
        for channel_id in channel_ids:
            mask[batch_id, channel_id] = 1

    return mask > 0


def generate_single_masks(shape, sampled_indexes):
    b, c, w, h = shape
    # Corrupt single values in multiple channels
    rand_c = torch.ones(c) > 0  # all channels
    rand_c = [idx for idx in range(c) if rand_c[idx]]
    rand_h, rand_w = torch.randint(0, h - 1, size=(1,)), torch.randint(
        0, w - 1, size=(1,)
    )
    mask = generate_4Dmask(shape, sampled_indexes, rand_c, rand_w, rand_w)
    return mask


def generate_line_masks(shape, sampled_indexes):
    b, c, w, h = shape
    # Corrupt rows or columns in multiple channels
    rand_line = torch.randint(high=h, size=(1,))
    rand_c = torch.bernoulli(torch.ones(c) * 0.75) > 0
    rand_c = [idx for idx in range(c) if rand_c[idx]]
    if torch.bernoulli(torch.ones(1) * 0.5):
        mask = generate_4Dmask(shape, sampled_indexes, rand_c, [], rand_line)
    else:
        mask = generate_4Dmask(shape, sampled_indexes, rand_c, rand_line, [])
    return mask


def generate_square_masks(shape, sampled_indexes):
    b, c, w, h = shape
    # Corrupt squares in multiple channels
    rand_c = torch.bernoulli(torch.ones(c) * 0.3) > 0
    if h - 1 == 0:
        h_0 = torch.tensor(0)
    else:
        h_0 = torch.randint(high=h - 1, size=(1,))
    h_1 = torch.randint(low=h_0.item(), high=h, size=(1,))
    if w - 1 == 0:
        w_0 = torch.tensor(0)
    else:
        w_0 = torch.randint(high=w - 1, size=(1,))
    w_1 = torch.randint(low=w_0.item(), high=w, size=(1,))
    rand_h = np.arange(h_0, h_1 + 1)
    rand_w = np.arange(w_0, w_1 + 1)
    mask = generate_4Dmask(shape, sampled_indexes, rand_c, rand_w, rand_h)
    return mask


def generate_all_masks(shape, sampled_indexes):
    b, c, w, h = shape
    # Corrupt entire channels
    rand_c = torch.bernoulli(torch.ones(c) * 0.1) > 0
    mask = generate_4Dmask(shape, sampled_indexes, rand_c)
    return mask


class HansGruberNI(torch.nn.Module):
    def __init__(
        self,
        args,
    ):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = args.error_model
        self.noise_data = list()
        self.p = (
            args.inject_p
        )  # fraction of the samples which the injection is applied to
        self.inject_epoch = (
            args.inject_epoch  # how many epochs before starting the injection
        )
        self.dummy_param = nn.Parameter(torch.empty(0))  # just to get the device
        self.mask_generators = [
            generate_line_masks,
            generate_square_masks,
            generate_all_masks,
        ]

    def set_noise_data(self, noise_data: list = None) -> None:
        r"""Set the noise data that we extract and parse from radiation experiments
        The noise data is extracted form a CSV file, pass only a numpy array to the function
        """
        # make a subset of the errors
        self.noise_data = [
            i for i in noise_data if i["geometry_format"] == self.error_model
        ]

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
            (1.0728769e-07, 1.0868737),
            (2.0230031, 1.0568325),
            (8.1847715e-08, 1.082071),
            (136027.72, 27.1194),
            (3.0, 1.0678725),
            (0.03517608, 1.189603),
            (3.4028237e38, 443107.0),
            (2.0, 1.4543958),
            (0.010238367, 1.1181921),
            (1.396856e-09, 1.0846596),
            (2.6865074e-10, 1.0769672),
            (1.3970158e-09, 1.085144),
            (0.66699225, 23.798765),
            (0.66699225, 23.798765),
            (0.66699225, 23.922783),
            (0.75000001, 121435080.0),
            (0.61141304, 3.4316596),
            (0.75000001, 121435080.0),
            (0.0, 1.08212),
            (7.0958774e-08, 1.082116),
            (0.0, 1.08212),
        ]

        alpha, x_min = random.choice(power_law_fus)
        r = random.random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        # print(relative_error)
        return relative_error
        # return 27.119592052269397

    def training_error(self, epoch=0):
        error = torch.rand(size=(1,), device=self.dummy_param.device) * max(1, epoch, 1)
        # error = torch.rand(size=(1,), device=self.dummy_param.device) * 6 + 1e-6
        if random.randint(0, 1):
            return error
        return -error

    def inject(
        self, forward_input: torch.Tensor, p: float, current_epoch: int = 0
    ) -> torch.Tensor:
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = forward_input.clone()
        linear = False
        try:
            b, c, h, w = output.shape
        except:
            b, c = output.shape
            linear = True

        # select the samples which the injection is applied to with probability p
        sampled_indexes = torch.bernoulli(torch.ones(b) * p)
        sampled_indexes = sampled_indexes > 0

        if self.training:
            error = self.training_error(current_epoch)
        else:
            error = self.random_relative_error

        if not self.training:
            # random
            if linear:
                mask = generate_2Dmask((b, c), sampled_indexes)
                output[mask] = output[mask].mul_(error)
            else:
                f = random.choice(self.mask_generators)
                mask = f(forward_input.shape, sampled_indexes)

        else:
            if linear:
                mask = generate_2Dmask((b, c), sampled_indexes)
                output[mask] = output[mask].mul_(error)
            else:
                if self.error_model == "single":
                    mask = generate_single_masks(forward_input.shape, sampled_indexes)

                elif self.error_model == "line":
                    mask = generate_line_masks(forward_input.shape, sampled_indexes)

                elif self.error_model == "square":
                    mask = generate_square_masks(forward_input.shape, sampled_indexes)

                elif self.error_model == "all":
                    mask = generate_all_masks(forward_input.shape, sampled_indexes)

                elif self.error_model == "random":
                    f = random.choice(self.mask_generators)
                    mask = f(forward_input.shape, sampled_indexes)

        output[mask] = output[mask].mul_(error)

        # check if the input has been modified
        # clean = torch.allclose(forward_input, output)
        return output, sampled_indexes

    def forward(
        self,
        forward_input: torch.Tensor,
        fwargs,
    ) -> torch.Tensor:
        r"""Perform a 'forward' operation to simulate the error model injection
        in the training process
        :param inject: whether to apply injection or not at test time
        :param forward_input: torch.Tensor input for the forward
        :return: processed torch.Tensor
        """
        output = forward_input
        if self.training:
            if fwargs["ep"] >= self.inject_epoch:
                # inject noise to each sample with probability p
                output, sampled_indexes = self.inject(forward_input, self.p)
        else:
            if fwargs["inj"]:
                # inject noise to all samples
                output, sampled_indexes = self.inject(forward_input, self.p)

        fwargs["faulty_idxs"] += (
            (fwargs["faulty_idxs"] < 0) * sampled_indexes * (fwargs["cnt"] + 1)
        )

        return output, fwargs
