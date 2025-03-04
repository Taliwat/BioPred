# This will be the script for Phase 1 that executes our other scripts to the Compute Cluster.
from azureml.core import Run, Workspace, Experiment, ScriptRunConfig, ComputeTarget
import json

# Use Run.get_context() to track execution
run = Run.get_context()

# Load the AzureML Workspace
workspace = Workspace.from_config(path = '/home/azureuser/cloudfiles/code/Users/kalpha1865/BioPred/Config/config.json')

# Attach compute cluster
compute_target = ComputeTarget(workspace = workspace, name = 'biopred-cluster-1')

# Define Experiment
experiment = Experiment(workspace, 'biopred_feature_engineering')

# Define Script Config
script_config = ScriptRunConfig(
    source_directory = '.',
    script = 'ph1_fe.py',
    compute_target = compute_target
)

# Submit job to Azure ML
print("Submitting job to Azure ML...")
run = experiment.submit(script_config)

# Wait for Completion and Stream Logs
run.wait_for_completion(show_output = True)

# Retrieve Logs and Metrics
metrics = run.get_metrics()
print("Logged Metrics:", metrics)
