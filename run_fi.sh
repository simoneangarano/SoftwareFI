#!/bin/bash

set -x
set -e
CFG_PATH=configurations

YAML_FILES=(  c100_resnet44_base_criticality.yaml
              c100_resnet44_injection_after_conv_criticality.yaml
              c10_resnet44_base_criticality.yaml
              c10_resnet44_full_injection_criticality.yaml )

DATA=data/reliability_evaluation
mkdir -p "$DATA"

for yaml_file in "${YAML_FILES[@]}"
do
  printf "%s\n" "$yaml_file"
  yaml_file_path="$CFG_PATH"/"$yaml_file"
  python3 ptl_to_torch.py -c "$yaml_file_path"
  randrange=1000
  injsite=neuron
  csv="$DATA"/fi_"$randrange"_"$injsite".csv
  python3 criticality_evaluation.py --config "$yaml_file_path" \
                                    --csv $csv \
                                    --injsite $injsite \
                                    --randrange $randrange
done