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

# We define a var for the path to the Meta checkpoint. Non-Googlers please remember to update the source `META_CHECKPOINT_PATH` to the GCS bucket where you have your Meta checkpoint
export META_CHECKPOINT_PATH=gs://scit1565-pedsllm-b5-def/Meta-Llama-3-70B-Instruct/

# In the following command, we are copying Meta's checkpoint into a local directory `tmp`. 
# You can use a different local directory than /tmp/, if you do so, please use the same local path for `base-model-path` when running `python3 MaxText/llama_or_mistral_ckpt.py`
#mkdir -p /tmp/meta-ckpt/
#gcloud storage cp -r ${META_CHECKPOINT_PATH} /tmp/meta-ckpt/

export BASE_OUTPUT_PATH=gs://scit1565-pedsllm-b5-def

echo "Converted checkpoints are stored at ${BASE_OUTPUT_PATH}"

export JAX_PLATFORMS=cpu
#Next, run the conversion script `MaxText/llama_or_mistral_ckpt.py` to convert Meta's PyTorch checkpoint in `base-model-path` and save the new converted (Orbax) checkpoint in the `maxtext-model-path`
/home/donatim/maxtext-md/MaxText/llama_or_mistral_ckpt.py --base-model-path "/home/donatim/Meta-Llama-3-70B-Instruct/" --maxtext-model-path ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/ --model-size ${MODEL_VARIATION}

echo "Wrote MaxText compatible checkpoint to ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt"

# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory.
#export CONVERTED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/0/items
# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `CONVERTED_CHECKPOINT` with `force_unroll=true`. 
#export RUN_NAME=unscanned_chkpt
#JAX_PLATFORMS=cpu python /home/donatim/maxtext-md/MaxText/generate_param_only_checkpoint.py /home/donatim/maxtext-md/MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${CONVERTED_CHECKPOINT} run_name=${RUN_NAME} model_name=${MODEL_VARIATION} force_unroll=true
#echo "Written MaxText compatible unscanned checkpoint to ${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items"

