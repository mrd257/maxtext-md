# Set the shell to use for commands
SHELL = /bin/bash

# Apply .ONESHELL:
.ONESHELL:

# Be sure to repeat any 

CONFIG_FILE=default_value
include $(CONFIG_FILE)

.PHONY: export_pod_envs
export_pod_envs:
	export PROJECT_ID=$(PROJECT_ID)
	export ACCELERATOR_TYPE=$(ACCELERATOR_TYPE)
