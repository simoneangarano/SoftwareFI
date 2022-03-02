#!/usr/bin/python3
import argparse
import time

import pandas as pd
import torch
import torchvision
from pytorchfi import core as pfi_core
from pytorchfi import neuron_error_models as pfi_neuron_error_models
from pytorchfi import weight_error_models as pfi_weight_error_models

from pytorch_scripts.utils import parse_args


def load_imagenet(data_dir: str, subset_size: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load imagenet from the folder <data_dir>/imagenet """
    # Get a dataset
    test_set = torchvision.datasets.imagenet.ImageNet(root=data_dir + "/imagenet", split="val", transform=transform)

    subset = torch.utils.data.Subset(test_set, range(subset_size))
    test_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    return test_loader


def load_cifar100(data_dir: str, transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load imagenet from the folder <data_dir>/imagenet """
    # Get a dataset
    test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=True, train=False,
                                                   transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader


def perform_fault_injection_for_a_model(args):
    model_path = "checkpoints/" + args.ckpt.replace("ckpt", "ts")
    min_val, max_val = -args.randrange, args.randrange
    inj_site = args.injsite
    csv_file = args.csv
    k = 5

    golden_model = torch.load(model_path)
    golden_model.eval()
    golden_model = golden_model.to("cuda")
    test_loader = load_cifar100(data_dir="data", transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]))
    # Testing PytorchFI
    pfi_model = pfi_core.fault_injection(golden_model, 1, input_shape=[3, 32, 32],
                                         layer_types=[
                                             torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                                             # torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                                             torch.nn.Linear],
                                         use_cuda=True)
    pfi_model.print_pytorchfi_layer_summary()
    sdc_counter, critical_sdc_counter = 0, 0
    injection_data = list()
    injected_faults = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image_gpu = image.to("cuda")
            # Golden execution
            model_time = time.time()
            gold_output = golden_model(image_gpu, inject=False)
            model_time = time.time() - model_time

            gold_output_cpu = gold_output.to("cpu")
            gold_top_k_labels = torch.topk(gold_output_cpu, k=k).indices.squeeze(0)
            gold_probabilities = torch.tensor(
                [torch.softmax(gold_output_cpu, dim=1)[0, idx].item() for idx in gold_top_k_labels])

            if inj_site == "neuron":
                inj = pfi_neuron_error_models.random_neuron_inj(pfi_model, min_val=min_val, max_val=max_val)
            elif inj_site == "weight":
                inj = pfi_weight_error_models.random_weight_inj(pfi_model, min_val=min_val, max_val=max_val)
            else:
                raise NotImplementedError("Only neuron and weight are supported as error models")

            inj = inj.eval()
            injection_time = time.time()
            inj_output = inj(image_gpu, inject=False)
            injection_time = time.time() - injection_time

            inj_output_cpu = inj_output.to("cpu")
            inj_top_k_labels = torch.topk(inj_output_cpu, k=k).indices.squeeze(0)
            inj_probabilities = torch.tensor(
                [torch.softmax(inj_output_cpu, dim=1)[0, idx].item() for idx in inj_top_k_labels])

            if i % 100 == 0:
                print(f"Time to gold {model_time} - Time to inject {injection_time}")
            injected_faults += 1
            if torch.any(torch.not_equal(gold_probabilities, inj_probabilities)):
                sdc, critical_sdc = 1, int(torch.any(torch.not_equal(gold_top_k_labels, inj_top_k_labels)))
                sdc_counter += sdc
                critical_sdc_counter += critical_sdc
                injection_data.append(
                    dict(SDC=sdc, critical_SDCs=critical_sdc,
                         gold_probs=gold_probabilities.tolist(), inj_probs=inj_probabilities.tolist(),
                         gold_labels=gold_top_k_labels.tolist(), inj_labels=inj_top_k_labels.tolist()))

            if i == 100:
                break
    injection_df = pd.DataFrame(injection_data)
    print(f"Injected faults {injected_faults} - SDC {sdc_counter} - Critical {critical_sdc_counter}")
    if csv_file:
        injection_df.to_csv(csv_file, index=False)
    else:
        print(injection_df)


def main() -> None:
    parser = config_parser = argparse.ArgumentParser(description='Criticality eval', add_help=False)
    parser.add_argument('--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments.')
    parser.add_argument('--randrange', default=100.0, type=float,
                        help="When injecting random values defines the range of the random values."
                             " [min_value=-randrange max_value=randrange]")
    parser.add_argument('--csv', default=None, type=str,
                        help="If this flag is enabled it will create a csv file --csv <path>")
    parser.add_argument('--injsite', default="neuron", help="Injection site, if pytorchfi can be neuron and weight")
    args = parse_args(parser, config_parser)
    perform_fault_injection_for_a_model(args)


if __name__ == '__main__':
    main()
