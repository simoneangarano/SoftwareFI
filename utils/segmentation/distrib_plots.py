from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from utils.utils import build_model

sns.set_theme(style="whitegrid")


def load_ptl_model(args):
    # Build model (Resnet only up to now)
    optim_params = {
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
    }
    n_classes = 10 if args.dataset == "cifar10" else 100

    ptl_model = build_model(args)
    checkpoint = torch.load(args.ckpt, map_location=torch.device("cpu"))
    ptl_model.load_state_dict(checkpoint["state_dict"])
    return ptl_model.model


def get_weights(model):
    weights = model.parameters()
    out = []

    for weight in weights:
        out += list(weight.flatten().detach().numpy())
    return out


def plot(df, idx=0):
    fig, ax = plt.subplots()
    # x = 'weights1' if idx == 0 else 'weights2'
    w = [weights1, weights2]
    colors = ["red", "green"]
    modes = ["Baseline", "Hardened"]
    plt.xlim(-0.5, 0.5)
    for x, c in zip(w, colors):
        sns.histplot(
            data=df,
            x=x,
            stat="frequency",
            binwidth=1e-2,
            color=c,
            alpha=0.4,
            ax=ax,
            legend=True,
        )
    fig.legend(modes, bbox_to_anchor=[0.915, 0.9], prop={"size": 14})
    plt.xlabel("Magnitude", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    fig.savefig("./dist_overlap.pdf")
    plt.show()


ckpt1 = "ckpt/c10_res44_test_01_bn-relu_base_sgd-epoch=99-val_acc=0.92.ckpt"
ckpt2 = "ckpt/c10_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.91.ckpt"
"""
ckpt1 = 'ckpt/c100_res44_test_01_bn-relu_base_sgd_9-epoch=99-val_acc=0.70.ckpt'
ckpt2 = 'ckpt/c100_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.69.ckpt'
"""

args = {
    "model": "hard_resnet44",
    "dataset": "cifar10",
    "loss": "bce",
    "inject_p": 0.0,
    "order": "bn-relu",
    "optimizer": "adamw",
    "epochs": 100,
    "lr": 2,
    "wd": 1e-3,
    "activation": "relu",
    "affine": True,
    "ckpt": ckpt1,
}
args = Namespace(**args)
model1 = load_ptl_model(args)
weights1 = get_weights(model1)

args.ckpt = ckpt2
model2 = load_ptl_model(args)
weights2 = get_weights(model2)


df = pd.DataFrame()
df["weights1"] = list(weights1)
df["weights2"] = list(weights2)
print(df.describe())

plot(df, 0)
# plot(df, 1)
