# SoftwareFI - Fault Injection for GhostNetV2

Based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897813.

## Directories

The directories are organized as follows:

* cfg - Configuration files for NVBITFI for fault injection
* utils - PyTorch scripts for training and inference for the used DNNs. For information, read
* utils/hg_noise_injector - A module to inject realistic errors in the training process

  the [README](/utils/README.md).

## Main script options

```bash
usage: main.py [-h] [--name NAME] [--mode MODE] [--ckpt CKPT] [--dataset DATASET] [--data_dir DATA_DIR] [--device DEVICE] [--loss LOSS] [--clip CLIP] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--optimizer OPTIMIZER] [--model MODEL] [--order ORDER]
               [--affine AFFINE] [--activation ACTIVATION] [--nan NAN] [--error_model ERROR_MODEL] [--inject_p INJECT_P] [--inject_epoch INJECT_EPOCH] [--wd WD] [--rand_aug RAND_AUG] [--rand_erasing RAND_ERASING] [--mixup_cutmix MIXUP_CUTMIX] [--jitter JITTER]
               [--label_smooth LABEL_SMOOTH] [--seed SEED] [--comment COMMENT]

PyTorch Training

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Experiment name.
  --mode MODE           Mode: train/training or validation/validate.
  --ckpt CKPT           Pass the name of a checkpoint to resume training.
  --dataset DATASET     Dataset name: cifar10 or cifar100.
  --data_dir DATA_DIR   Path to dataset.
  --device DEVICE       Device number.
  --loss LOSS           Loss: bce, ce or sce.
  --clip CLIP           Gradient clipping value.
  --epochs EPOCHS       Number of epochs.
  --batch_size BATCH_SIZE
                        Batch Size
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer name: adamw or sgd.
  --model MODEL         Network name. Resnets only for now.
  --order ORDER         Order of activation and normalization: bn-relu or relu-bn.
  --affine AFFINE       Whether to use Affine transform after normalization or not.
  --activation ACTIVATION
                        Non-linear activation: relu or relu6.
  --nan NAN             Whether to convert NaNs to 0 or not.
  --error_model ERROR_MODEL
                        Optimizer name: adamw or sgd.
  --inject_p INJECT_P   Probability of noise injection at training time.
  --inject_epoch INJECT_EPOCH
                        How many epochs before starting the injection.
  --wd WD               Weight Decay.
  --rand_aug RAND_AUG   RandAugment magnitude and std.
  --rand_erasing RAND_ERASING
                        Random Erasing propability.
  --mixup_cutmix MIXUP_CUTMIX
                        Whether to use mixup/cutmix or not.
  --jitter JITTER       Color jitter.
  --label_smooth LABEL_SMOOTH
                        Label Smoothing.
  --seed SEED           Random seed for reproducibility.
  --comment COMMENT     Optional comment.

```

# To cite the original work

**2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)**

```bibtex
@INPROCEEDINGS{diehardnetIOLTS2022,
  author={Cavagnero, Niccolò and Santos, Fernando Dos and Ciccone, Marco and Averta, 
          Giuseppe and Tommasi, Tatiana and Rech, Paolo},
  booktitle={2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)}, 
  title={Transient-Fault-Aware Design and Training to Enhance DNNs Reliability with Zero-Overhead}, 
  year={2022},
  pages={1-7},
  doi={10.1109/IOLTS56730.2022.9897813}
}

```

# Neutron beam evaluations

The setup files and scripts for validating with neutron beams are available at
[diehardnetradsetup](https://github.com/diehardnet/diehardnetradsetup)