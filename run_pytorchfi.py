#!/usr/bin/python3
import argparse
import time

import pandas as pd
import torch
import torchvision
from pytorchfi import core as pfi_core
from pytorchfi import neuron_error_models as pfi_neuron_error_models

from utils.utils import build_model, parse_args
from utils.test_utils import CoreDataModule


# single random neuron error in single batch element
def random_neuron_inj(pfi, min_val=-1, max_val=1):
    b = pfi_neuron_error_models.random_batch_element(pfi)
    (layer, C, H, W) = pfi_neuron_error_models.random_neuron_location(pfi)
    err_val = pfi_neuron_error_models.random_value(min_val=min_val, max_val=max_val)
    pfi_obj = pfi.declare_neuron_fi(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )
    config = dict(layer=layer, channel=C, height=H, width=W, err_val=err_val)
    return pfi_obj, config


def load_ptl_model(args):
    # Build model (Resnet only up to now)
    optim_params = {
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
    }
    n_classes = 1000

    ptl_model = build_model(
        model=args.model,
        n_classes=n_classes,
        optim_params=optim_params,
        loss=args.loss,
        inject_p=args.inject_p,
        order=args.order,
        activation=args.activation,
        affine=args.affine,
    )

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, weights_only=True)
        ptl_model.load_state_dict(checkpoint["state_dict"])
    return ptl_model.model


def perform_fault_injection_for_a_model(args):
    min_val, max_val = -args.randrange, args.randrange
    inj_site = args.injsite
    csv_file = args.csv
    k = 5

    # golden_model = torch.load(model_path)
    golden_model = load_ptl_model(args=args)
    golden_model.eval()
    golden_model = golden_model.to("cuda")

    # Load the dataset
    datamodule = CoreDataModule(batch_size=128)
    test_loader = datamodule.test_dataloader()

    # Testing PytorchFI
    pfi_model = pfi_core.fault_injection(
        golden_model,
        1,
        input_shape=[5, 512, 512],
        layer_types=[
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.Linear,
        ],
        use_cuda=True,
    )
    pfi_model.print_pytorchfi_layer_summary()
    sdc_counter, critical_sdc_counter = 0, 0
    injection_data = list()
    injected_faults = 0
    total_time = time.time()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image_gpu = image.to("cuda")
            # Golden execution
            model_time = time.time()
            out, intermediates = golden_model(image_gpu, inject=False)
            model_time = time.time() - model_time
            gold_output = intermediates[-1]
            gold_output_cpu = gold_output.to("cpu")
            # gold_top_k_labels = torch.topk(gold_output_cpu, k=k).indices.squeeze(0)
            # gold_probabilities = torch.tensor(
            #     [
            #         torch.softmax(gold_output_cpu, dim=1)[0, idx].item()
            #         for idx in gold_top_k_labels
            #     ]
            # )

            if inj_site == "neuron":
                inj, config_dict = random_neuron_inj(
                    pfi_model, min_val=min_val, max_val=max_val
                )
            else:
                raise NotImplementedError(
                    "Only neuron and weight are supported as error models"
                )

            inj.eval()
            injection_time = time.time()
            out, intermediates = inj(image_gpu, inject=False)
            injection_time = time.time() - injection_time
            inj_output = intermediates[-1]
            inj_output_cpu = inj_output.to("cpu")
            # inj_top_k_labels = torch.topk(inj_output_cpu, k=k).indices.squeeze(0)
            # inj_probabilities = torch.tensor(
            #     [
            #         torch.softmax(inj_output_cpu, dim=1)[0, idx].item()
            #         for idx in inj_top_k_labels
            #     ]
            # )

            if i % 100 == 0:
                print(f"Time to gold {model_time} - Time to inject {injection_time}")
            injected_faults += 1

            # gold_top1_label = int(torch.topk(gold_output, k=1).indices.squeeze(0))
            # inj_top1_label = int(torch.topk(inj_output, k=1).indices.squeeze(0))
            # gold_top1_prob = torch.softmax(gold_output, dim=1)[
            #     0, gold_top1_label
            # ].item()
            # inj_top1_prob = torch.softmax(inj_output, dim=1)[0, inj_top1_label].item()

            # if torch.any(torch.not_equal(gold_probabilities, inj_probabilities)):
            #     sdc, critical_sdc = 1, int(
            #         torch.any(torch.not_equal(gold_top_k_labels, inj_top_k_labels))
            #     )
            #     sdc_counter += sdc
            #     critical_sdc_counter += critical_sdc
            injection_data.append(
                dict(
                    mse=torch.nn.functional.mse_loss(gold_output, inj_output).item(),
                    mae=torch.nn.functional.l1_loss(gold_output, inj_output).item(),
                    # SDC=sdc,
                    # critical_SDCs=critical_sdc,
                    # gold_probs=gold_probabilities.tolist(),
                    # inj_probs=inj_probabilities.tolist(),
                    # gold_labels=gold_top_k_labels.tolist(),
                    # inj_labels=inj_top_k_labels.tolist(),
                    # ground_truth_label=label,
                    # gold_top1_label=gold_top1_label,
                    # inj_top1_label=inj_top1_label,
                    # gold_top1_prob=gold_top1_prob,
                    # inj_top1_prob=inj_top1_prob,
                    **config_dict,
                    # gold_argmax=torch.max(gold_output_cpu, 1), inj_argmax=torch.max(inj_output_cpu, 1)
                )
            )

            # if i == 100:
            #     break
    injection_df = pd.DataFrame(injection_data)
    print(
        f"Injected faults {injected_faults} - SDC {sdc_counter} - Critical {critical_sdc_counter}"
    )
    total_time = time.time() - total_time
    print(f"{total_time:.2f}")
    if csv_file:
        injection_df["injected_faults"] = injected_faults
        injection_df.to_csv(csv_file, index=False)
    else:
        print(injection_df)


def main() -> None:
    parser = config_parser = argparse.ArgumentParser(
        description="Criticality eval", add_help=False
    )
    parser.add_argument(
        "--config",
        default="cfg/ghostnetv2.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments.",
    )
    args = parse_args(parser, config_parser)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()
    perform_fault_injection_for_a_model(args)


if __name__ == "__main__":
    main()
