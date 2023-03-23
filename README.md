# DieHardNET

Repository for a reliable Deep Neural Network (DNN) model. DieHardNet stands for
**Die** (integrated circuit) **Har**dened Neural **Net**work

[comment]: <> (TODO: Replace by two images from john mcclane one 
classified with DieHardNet and other with an error
 l![Die hard photo]&#40;/diehard.jpg&#41;)

## Directories

The directories are organized as follows:

* hg_noise_injector - A module to inject realistic errors in the training process
* eval_fault_injection_cfg - Configuration files for NVBITFI for fault injection
* pytorch_scripts - PyTorch scripts for training and inference for the used DNNs. For information read
  the [README](/pytorch_scripts/README.md).

## Run the training process

<ol>
<li>With the Hans Gruber error injector</li>

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

<li>Without the error injector</li>

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

</ol>

## Run the inference

An example of running the inference with the pretrained hardened model

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

# To cite this work

**2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)**

```bibtex
@INPROCEEDINGS{diehardnetIOLTS2022,
  author={Cavagnero, Niccolò and Santos, Fernando Dos and Ciccone, Marco and Averta, Giuseppe and Tommasi, Tatiana and Rech, Paolo},
  booktitle={2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)}, 
  title={Transient-Fault-Aware Design and Training to Enhance DNNs Reliability with Zero-Overhead}, 
  year={2022},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IOLTS56730.2022.9897813}
}

```
