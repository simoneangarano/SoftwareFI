#!/bin/bash

# Standard ResNet w/ Relu and Relu6. No noise aware training.
python3 main.py -c configurations/c10_res44_test_01_bn-relu_base.yaml
#python3 main.py -c configurations/c10_res44_test_01_bn-relu_base_adamw.yaml

python3 main.py -c configurations/c10_res44_test_02_bn-relu6_base.yaml
#python3 main.py -c configurations/c10_res44_test_02_bn-relu6_base_adamw.yaml

# Standard ResNet w/ Relu and Relu6. Noise aware training.
#python3 main.py -c configurations/c10_res44_test_01_bn-relu.yaml
#python3 main.py -c configurations/c10_res44_test_02_bn-relu6.yaml

# Standard ResNet w/ inverted act-bn, w/ Relu and Relu6. No noise aware training.
#python3 main.py -c configurations/c10_res44_test_01_relu-bn_base.yaml
python3 main.py -c configurations/c10_res44_test_02_relu6-bn_base.yaml
#python3 main.py -c configurations/c10_res44_test_02_relu6-bn_base_adamw.yaml

# Standard ResNet w/ inverted act-bn, w/ Relu and Relu6. Noise aware training.
#python3 main.py -c configurations/c10_res44_test_01_relu-bn.yaml
python3 main.py -c configurations/c10_res44_test_02_relu6-bn.yaml
#python3 main.py -c configurations/c10_res44_test_02_relu6-bn_adamw.yaml
