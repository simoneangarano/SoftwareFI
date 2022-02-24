#!/bin/bash

python3 main.py -c configurations/c10_resnet20_base_full_injection.yaml
python3 main.py -c configurations/c10_resnet32_base_full_injection.yaml
python3 main.py -c configurations/c10_resnet44_base_full_injection.yaml
python3 main.py -c configurations/c100_resnet20_base_full_injection.yaml
python3 main.py -c configurations/c100_resnet32_base_full_injection.yaml
python3 main.py -c configurations/c100_resnet44_base_full_injection.yaml