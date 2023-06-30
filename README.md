# Production ML Deployment
 This repository is part 2 of a series of templates for stepping up a POC deployment to one that is scalable in production. This part introduces:
 1. training pipeline and production pipelines
 2. two critical tools to complement the above pipelines:  
 &nbsp; a. Optuna - parallelized hyperparameter search  
 &nbsp; b. MLFlow - experiment tracking and model registry

## Background
Architecture...
#### 1. Training and Production pipelines
On the road to a production level deployment, we define two types of pipelines: training and production. At the front end of the ML workflow (e.g. data cleaning and feature engineering) they remain the same for both. The main difference is that training pipelines will do a 70/30 split (or whatever split you'd like) of the data for searching optimal hyperparameters, whereas production pipelines will load the optimal hyperparameters and use 100% of the data to build and register the model for inference. Inference will be done using FastAPI, as seen in our part 1. 
![](/docs/training-production.jpg)
Note: we will not be doing any model monitoring or dynamic retraining of the model. This production workflow is at level 0 automation (manual monitoring and retraining).

#### 2a. Optuna for parallelized hyperparameter search

#### 2b. MLFlow for Experiment Tracking and Model Registry 

Our local MLflow server will be built via Docker. For it's architecture, it is connected to Postgres for experiment tracking and Minio for object storage. This can be easily configured within the docker-compose.yaml file if you'd like to swap out your RDBMS or object store. 


## Usage
#### Start MlFlow server via Docker
```
docker-compose build --no-cache
docker-compose up --remove-orphans --force-recreate -d
```