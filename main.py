# implements the PytorchEstimator

import json
import boto3
import time
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()


# The bucket containig our input data
bucket = 's3://mr-testing-oneoff'


# The IAM Role which SageMaker will impersonate to run the estimator
# Remember you cannot use sagemaker.get_execution_role()
# if you're not in a SageMaker notebook, an EC2 or a Lambda
# (i.e. running from your local PC)
role = sagemaker.get_execution_role()


output_path = f'{bucket}/sagemaker-jobs'
# Set the job name and show it
trial_name = "experiment-V0"
job_name = "torch-spot-{}-{}".format(trial_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
print(f"Job name: {job_name}")
output_jobs_path = f'{output_path}/{job_name}'
checkpoint_s3_uri = f'{output_jobs_path}/checkpoints'
tb_config = TensorBoardOutputConfig(
    s3_output_path=f"{output_jobs_path}/tensorboard",
#     container_local_output_path="/opt/ml/output/tensorboard",
)


# The instance to train on
framework_version = "1.6.0" # 1.7.1
instance_type = 'ml.p3.8xlarge'  # The library supports ml.p3.16xlarge, ml.p3dn.24xlarge, and ml.p4d.24xlarge instances at this time.
gpus_per_host = 4  # Must relate to instance type, see https://aws.amazon.com/ec2/instance-types/p3/
instance_count = 2
volume_size = 2 # Number of Gb disc to use, for this example we dont really need any...
max_run = 5 # Minutes  

# distributed options, not really necessary...
mpi_options = {
    "enabled": True,
    "processes_per_host": gpus_per_host,
    "custom_mpi_options":"-verbose --NCCL_DEBUG=INFO --mca btl_vader_single_copy_mechanism none"
  }
#     "custom_mpi_options" : "--mca btl_vader_single_copy_mechanism none "

# SDP distribution method enabled here
# smdistributed_options = {
#     "dataparallel": {
#         "enabled": True
#     }
# }

distributions = {
    "mpi": mpi_options,
#     "smdistributed": smdistributed_options # Dont use this yet not setup...
}


# Model training hyperparameters
hyperparameters = {
    # Trainer parameters
    'max_epochs': 50,
    'batch_size': 64,
    "nsamples": 50000,
    "auto_lr_find": True,
    "gradient_clip_val": 2,
    "num_nodes": instance_count,

    # set the distributed params
    'profiler': True,
    "accelerator": "ddp",
    #"plugins": "ddp_sharded",

    # DataModule
    #"num_workers": 4
}


# Creates a new PyTorch Estimator with params
estimator = PyTorch(
    output_path=output_jobs_path,
    # name of the runnable script containing __main__ function (entrypoint)
    entry_point='train.py',
    # path of the folder containing training code. It could also contain a
    # requirements.txt file with all the dependencies that needs
    # to be installed before running
    source_dir='code',
    py_version='py3',
    role=role,
    framework_version=framework_version,
    instance_count=instance_count,
    instance_type=instance_type,
    # these hyperparameters are passed to the main script as arguments and
    # can be overridden when fine tuning the algorithm
    hyperparameters=hyperparamters,
    use_spot_instances=True,
    max_wait=max_run * 60,
    max_run=max_run * 60,
    # Now set checkpoint s3 path, local default this will be /opt/ml/checkpoints/
    checkpoint_s3_uri=checkpoint_s3_uri,
    tensorboard_output_config=tb_config,
    distributions=distributions,
    volume_size=volume_size,
)


# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container
estimator.fit()
