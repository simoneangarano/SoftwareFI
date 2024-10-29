import json
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from utils.data.data_module import CoreDataModule
from utils.models.ghostnetv2 import *
from utils.utils import ModelWrapper, RunningStats, get_parser, parse_args

INJECT = True

LAYERS = [
    ConvInjector,
    BNInjector,
    NaNAct,
    ClampAvgPool2d,
]

ACTIVATIONS, STATS = {}, {}


def main(args):
    datamodule = CoreDataModule(args, batch_size=args.batch_size, drop_last=True)

    # Define the model
    backbone = ghostnetv2(
        inject=INJECT,
        error_model=args.error_model,
        inject_p=args.inject_p,
        inject_epoch=args.inject_epoch,
    )
    head = SegmentationHeadGhostBN(inject=INJECT)
    model = GhostNetSS(backbone, head)
    model = load_fi_weights(model, args.ckpt).cuda()
    model = ModelWrapper(model, args.num_classes, loss=args.loss)

    get_stats(model, datamodule, args, inject=INJECT)

    results = {key: val.get_stats() for key, val in STATS.items()}
    json.dump(results, open(f"ckpt/{args.name}_stats_{INJECT}.json", "w"))


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

        POOL = ThreadPool(8)

        def update_stats(inputs):
            name, out = inputs
            STATS[name].push(out.cpu().numpy())

        POOL.map(update_stats, ACTIVATIONS.items())
        POOL.close()
        POOL.join()


@torch.no_grad()
def get_activation(name):
    def hook(model, input, output):
        ACTIVATIONS[name] = output[0].detach()

    return hook


if __name__ == "__main__":

    parser, config_parser = get_parser()
    args = parse_args(parser, config_parser, args="", verbose=False)

    main(args)
