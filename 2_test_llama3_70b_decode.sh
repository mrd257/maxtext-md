#!/bin/bash

# This file is both an integration test that runs once a day on a v4-16 and documentation for how to get started with Llama3-8b. 
# Please make sure you have run end_to_end/tpu/llama3/8b/1_test_llama3_8b.sh before running commands from this file. 

# The flow of this file is as follows:
# 1. Run decoding, finetuning of Llama3-8B with the converted checkpoint obtained from end_to_end/tpu/llama3/8b/1_test_llama3_8b.sh. Also, run pretraining of Llama3-8B
# 2. Run more efficient decoding with the unscanned checkpoint obtained from end_to_end/tpu/llama3/8b/1_test_llama3_8b.sh
# 3. Run decoding from the finetuned checkpoint from step 1


# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/llama3/8b/2_test_llama3_8b.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/llama3/8b/1_test_llama3_8b.sh
# Please note that in these two scripts (1_test_llama3_8b.sh and 2_test_llama3_8b.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex
export MODEL_VARIATION='llama3-70b'

export BASE_OUTPUT_PATH=gs://scit1565-pedsllm-b5-def

# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://maxtext-dataset

# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory. This way it is easier to use this path in the `train.py` and `decode.py` commands
export CONVERTED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/0/items
export RUN_NAME=unscanned_chkpt
# We defined path to unscanned checkpoint created in 1_test_llama3_8b.sh
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items

# We run decoding on the `UNSCANNED_CKPT_PATH` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state. 
# So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
#python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false scan_layers=false model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"

# We can also run decoding (albeit in a bit unoptimized way) by using the scanned converted checkpoint located at `CONVERTED_CHECKPOINT`. Note again that this checkpoint only has parameters and no optimizer state. So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python3 MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"

# We also test whether the forward pass logits match the golden logits for Llama3-8B
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic dtype=float32 async_checkpointing=false scan_layers=false
