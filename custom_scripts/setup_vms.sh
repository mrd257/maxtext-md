

export_single_host_envs(){
    # be sure to change user id in TPU_NAME 
    export PROJECT_ID=scit1565-pedsllm-b5 
    export ACCELERATOR_TYPE=v5litepod-8
    export ZONE=us-central1-a 
    export RUNTIME_VERSION=v2-alpha-tpuv5-lite
    export SERVICE_ACCOUNT=scit1565-pedsllm-b5@scit1565-pedsllm-b5.iam.gserviceaccount.com 
    export TPU_NAME=donatim-v5litepod-8
    export QUEUED_RESOURCE_ID=$TPU_NAME
    export NETWORK=projects/arcus-vpc-master/global/networks/arcus-master-vpc	\
    export SUBNETWORK=projects/arcus-vpc-master/regions/us-central1-a/subnetworks/scit1565-pedsllm-b5-central1
    export QUEUED_RESOURCE_NAME=projects/scit1565-pedsllm-b5/locations/us-central1-a/queuedResources/donatim-v5litepod-8
    gcloud config set project $PROJECT_ID --quiet
    gcloud config set compute/zone $ZONE --quiet
}

export_multihost_16_envs(){
    export PROJECT_ID=scit1565-pedsllm-b5 
    export ACCELERATOR_TYPE=v5litepod-16
    export ZONE=us-south1-a 
    export RUNTIME_VERSION=v2-alpha-tpuv5-lite
    export SERVICE_ACCOUNT=scit1565-pedsllm-b5@scit1565-pedsllm-b5.iam.gserviceaccount.com 
    export TPU_NAME=donati-v5litepod-16
    export QUEUED_RESOURCE_ID=$TPU_NAME
    export NETWORK=projects/arcus-vpc-master/global/networks/arcus-master-vpc	\
    export SUBNETWORK=projects/arcus-vpc-master/regions/us-south1-a/subnetworks/scit1565-pedsllm-b5-south1
    export QUEUED_RESOURCE_NAME=projects/scit1565-pedsllm-b5/locations/${ZONE}/queuedResources/${TPU_NAME}
    gcloud config set project $PROJECT_ID --quiet
    gcloud config set compute/zone $ZONE --quiet
}

export_multihost_64_envs(){
    export PROJECT_ID=scit1565-pedsllm-b5 
    export ACCELERATOR_TYPE=v5litepod-64
    export ZONE=us-south1-a 
    export RUNTIME_VERSION=v2-alpha-tpuv5-lite
    export SERVICE_ACCOUNT=scit1565-pedsllm-b5@scit1565-pedsllm-b5.iam.gserviceaccount.com 
    export TPU_NAME=donati-v5litepod-64
    export QUEUED_RESOURCE_ID=$TPU_NAME
    export NETWORK=projects/arcus-vpc-master/global/networks/arcus-master-vpc	\
    export SUBNETWORK=projects/arcus-vpc-master/regions/us-south1-a/subnetworks/scit1565-pedsllm-b5-south1
    export QUEUED_RESOURCE_NAME=projects/scit1565-pedsllm-b5/locations/${ZONE}/queuedResources/${TPU_NAME}
    gcloud config set project $PROJECT_ID --quiet
    gcloud config set compute/zone $ZONE --quiet
}

export_multihost_128_envs(){
    export PROJECT_ID=scit1565-pedsllm-b5 
    export ACCELERATOR_TYPE=v5litepod-128
    export ZONE=us-south1-a 
    export RUNTIME_VERSION=v2-alpha-tpuv5-lite
    export SERVICE_ACCOUNT=scit1565-pedsllm-b5@scit1565-pedsllm-b5.iam.gserviceaccount.com 
    export TPU_NAME=donati-v5litepod-128
    export QUEUED_RESOURCE_ID=$TPU_NAME
    export NETWORK=projects/arcus-vpc-master/global/networks/arcus-master-vpc	\
    export SUBNETWORK=projects/arcus-vpc-master/regions/us-south1-a/subnetworks/scit1565-pedsllm-b5-south1
    export QUEUED_RESOURCE_NAME=projects/scit1565-pedsllm-b5/locations/${ZONE}/queuedResources/${TPU_NAME}
    gcloud config set project $PROJECT_ID --quiet
    gcloud config set compute/zone $ZONE --quiet
}

# Need to pass in "--best-effort" as argument if want to use spot requisition
requisition_resources(){
    if [[ "$#" -eq 1 && "$1" == "--best-effort" ]]; then 
        best_effort_flag="$1"
    fi
    echo $best_effort_flag

    gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
        --node-id ${TPU_NAME} \
        --project ${PROJECT_ID} \
        --network ${NETWORK} \
        --subnetwork ${SUBNETWORK} \
        --zone ${ZONE} \
        --accelerator-type ${ACCELERATOR_TYPE} \
        --runtime-version ${RUNTIME_VERSION} \
        --service-account ${SERVICE_ACCOUNT} \
        --internal-ips \
        ${best_effort_flag:+"$best_effort_flag"}

    # Loop until the status is ACTIVE"
    while true; do
        status=$(gcloud compute tpus queued-resources list --filter=name=${QUEUED_RESOURCE_NAME} | awk 'NR==2 {print $5}')
        if echo "$status" | grep -q "ACTIVE"; then
            echo "Host is active. Proceeding to next command."
            break
        else
            echo $status
            echo "Waiting for host to become active..."
        fi
        sleep 5
    done
}


release_resources(){
    gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE} --quiet
    gcloud compute tpus queued-resources delete ${TPU_NAME}
}

ssh_to_host(){
    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE  --internal-ip        
}

