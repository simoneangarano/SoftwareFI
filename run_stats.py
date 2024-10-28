import json

import torch
from tqdm import tqdm

from utils.data.data_module import CoreDataModule
from utils.models.ghostnetv2 import *
from utils.utils import ModelWrapper, RunningStats, get_parser, parse_args

LAYERS = [
    ConvInjector,
    BNInjector,
    NaNAct,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AvgPool2d,
]

ACTIVATIONS, STATS = {}, {}


def main(args):
    datamodule = CoreDataModule(args, batch_size=args.batch_size)

    # Define the model
    backbone = ghostnetv2(
        inject=False,
        error_model=args.error_model,
        inject_p=args.inject_p,
        inject_epoch=args.inject_epoch,
    )
    head = SegmentationHeadGhostBN(inject=False)
    model = GhostNetSS(backbone, head)
    model = load_fi_weights(model, args.ckpt).cuda()
    model = ModelWrapper(model, args.num_classes, loss=args.loss)

    get_stats(model, datamodule, args, inject=False)

    results = {key: val.get_stats() for key, val in STATS.items()}
    json.dump(results, open(f"ckpt/{args.name}_stats.json", "w"))


@torch.no_grad()
def get_stats(net: ModelWrapper, datamodule, args, inject=False):

    for name, layer in net.named_modules():
        if any(isinstance(layer, t) for t in LAYERS):
            layer.register_forward_hook(get_activation(name))
            STATS[name] = RunningStats()

    for _, batch in tqdm(enumerate(datamodule.val_dataloader())):
        batch = [b.cuda() for b in batch]
        x, _ = batch
        _ = net(x, inject=inject, inject_index=args.inject_index)
        for name, out in ACTIVATIONS.items():
            STATS[name] += out.cpu().numpy()


def get_activation(name):
    def hook(model, input, output):
        ACTIVATIONS[name] = output[0].detach()

    return hook


if __name__ == "__main__":

    parser, config_parser = get_parser()
    args = parse_args(parser, config_parser, args="", verbose=False)

    main(args)
