import json
import numpy as np
import random
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.data.data_module import CoreDataModule
from utils.models.ghostnetv2 import *
from utils.utils import ModelWrapper, RunningStats, get_parser, parse_args, build_model


LAYERS = [
    ConvInjector,
    BNInjector,
    NaNAct,
    ClampAvgPool2d,
]

ACTIVATIONS, STATS = {}, {}


def main(args):
    datamodule = CoreDataModule(args)

    # Define the model
    net = build_model(args)
    net = net.cuda().eval()

    get_stats(net, datamodule, args, inject=args.inject)

    results = {key: val.get_stats() for key, val in STATS.items()}
    json.dump(results, open(f"ckpt/{args.exp}_stats.json", "w"))


@torch.no_grad()
def get_stats(net: ModelWrapper, datamodule, args, inject=False):
    for name, layer in net.named_modules():
        if any(isinstance(layer, t) for t in LAYERS):
            layer.register_forward_hook(get_activation(name))
            STATS[name] = RunningStats()
    STATS["logits"] = RunningStats()

    for _, batch in tqdm(enumerate(datamodule.dataloader())):
        batch = [b.cuda() for b in batch]
        x, _ = batch
        outputs = net(x, inject=inject, inject_index=args.inject_index)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        ACTIVATIONS["logits"] = outputs

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            def update_stats(inputs):
                name, out = inputs
                STATS[name].push(out.cpu().numpy())

            futures = [
                executor.submit(update_stats, item) for item in ACTIVATIONS.items()
            ]
            for future in futures:
                future.result()  # Wait for all threads to complete


@torch.no_grad()
def get_activation(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            output = output[0]
        ACTIVATIONS[name] = output.detach()

    return hook


if __name__ == "__main__":

    parser, config_parser = get_parser()
    args = parse_args(parser, config_parser, args="", verbose=True)

    # Avoid memory issues
    args.split = "test"
    args.batch_size //= 8
    args.drop_last = True
    args.stats = False
    args.detect = False

    # Set random seed
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
