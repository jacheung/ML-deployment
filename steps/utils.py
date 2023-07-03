import mlflow

# mlflow Tracking requires definition of experiment name AND logged params
# Experiment names they should be defined as "project-task-version"
def set_mlflow_experiment(experiment_name:str, artifact_location: str = None):
    try:
        experiment_id = mlflow.create_experiment(experiment_name, 
                                                 artifact_location=artifact_location)
    # except mlflow.exceptions.MlflowException as e:
    #   if str(e) == f"Experiment '{experiment_name}' already exists.":
    except:
        print(f'Experiment already exists, setting experiment to {experiment_name}')
        experiment_info = mlflow.set_experiment(experiment_name)
        experiment_id = experiment_info.experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    print("---------------------")
    print('Experiment details are:')
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Creation timestamp: {}".format(experiment.creation_time))
    return experiment_id
