# production-ML-deployment
 This repository is part 2 of a series of templates for stepping up a POC deployment to one that is scalable in production. This part introduces:
 1. training/research pipeline and production pipelines  
 2. two critical tools to complement the above pipelines:  
 &nbsp; a. Optuna - parallelized hyperparameter search  
 &nbsp; b. MLFlow - experiment tracking and model registry

## 1. Training and Production pipelines

## 2a. Optuna for parallelized hyperparameter search

## 2b. MLFlow for Experiment Tracking and Model Registry 

Our local MLflow server will be built via Docker. For it's architecture, it is connected to Postgres for experiment tracking and Minio for object storage. This can be easily configured within the docker-compose.yaml file if you'd like to swap out your RDBMS or object store. 

#### Start MlFlow server via Docker
```
docker-compose build --no-cache
docker-compose up --remove-orphans --force-recreate -d
```