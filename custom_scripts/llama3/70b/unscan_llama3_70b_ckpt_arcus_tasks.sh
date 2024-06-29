#!/bin/bash

# This file, combined with step 2 in the same directory, demonstrates converting a Llama3-8B checkpoint from Meta and running various MaxText operations on it.
# This step is tested nightly on an ordinary CPU VM.

# The flow of this file is as follows:
# 1. Pull the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/llama3/8b/1_test_llama3_8b.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/llama3/8b/2_test_llama3_8b.sh.
# Please note that in these two scripts (1_test_llama3_8b.sh and 2_test_llama3_8b.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex
MODEL_VARIATION='llama3-70b'

export BASE_OUTPUT_PATH=gs://scit1565-pedsllm-b5-def

# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory.
export CONVERTED_CHECKPOINT=gs://scit1565-pedsllm-b5-def/llama3-70b/scanned_chkpt/0/items
# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `CONVERTED_CHECKPOINT` with `force_unroll=true`. 
export RUN_NAME=unscanned_chkpt
export JAX_PLATFORMS=cpu
conda run --no-capture-output -n convert-checkpoint python /home/donatim/maxtext-md/MaxText/generate_param_only_checkpoint.py /home/donatim/maxtext-md/MaxText/configs/base.yml async_checkpointing=false base_output_directory=gs://scit1565-pedsllm-b5-def load_parameters_path=gs://scit1565-pedsllm-b5-def/llama3-70b/scanned_chkpt/0/items run_name=unscan70b model_name='llama3-70b' force_unroll=true

