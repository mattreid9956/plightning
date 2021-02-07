# implements the PytorchEstimator

import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()


# The bucket containig our input data
bucket = 's3://deep-q-captial-data'


# The IAM Role which SageMaker will impersonate to run the estimator
# Remember you cannot use sagemaker.get_execution_role()
# if you're not in a SageMaker notebook, an EC2 or a Lambda
# (i.e. running from your local PC)
role = sagemaker.get_execution_role()


# The bucket containig our input data
bucket = "s3://MY_TEST_BUCKET"
output_path = f'{bucket}/sagemaker-jobs'
# Set the job name and show it
trial_name = "dqc-V0-btcusd-coinbase"
job_name = "torch-spot-{}-{}".format(trial_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
print(f"Job name: {job_name}")
output_jobs_path = f'{output_path}/{job_name}'
checkpoint_s3_uri = f'{output_jobs_path}/checkpoints'
tb_config = TensorBoardOutputConfig(
    s3_output_path=f"{output_jobs_path}/tensorboard",
    container_local_output_path="/opt/ml/output/tensorboard",
)


# The instance to train on
framework_version = "1.6.0"
instance_type = 'ml.p3.16xlarge'
instance_count = 2


# Creates a new PyTorch Estimator with params
estimator = PyTorch(
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
    hyperparameters={
        # Trainer parameters
        'max_epochs': 50,
        'batch_size': 8,
        "auto_lr_find": True,
        "gradient_clip_val": 2,
        
        # set the distributed params
        'profiler': True,
        "accelerator": "ddp",
        "plugins": "ddp_sharded",

        # DataModule
        #"num_workers": 4,        
    },
    #use_spot_instances=True,
    #max_wait=600,
    # Now set checkpoint s3 path, local default this will be /opt/ml/checkpoints/
    checkpoint_s3_uri=checkpoint_s3_uri,
    distribution={'smdistributed':{'dataparallel':{enabled': True}}}
)


# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container
estimator.fit()
