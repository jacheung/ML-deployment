# Production ML Deployment
 This repository is part 2 of a series of templates for stepping up a POC deployment to one that is scalable in production. This part introduces:
 1. training pipeline and production pipelines
 2. two critical tools to complement the above pipelines:  
 &nbsp; a. Optuna - parallelized Bayesian hyperparameter search  
 &nbsp; b. MLflow - experiment tracking and model registry

![](/docs/vision-production.png) 

## Background
We'll be performing transfer learning on the MNIST dataset to demo the tools used in this production level workflow. The architecture diagram below will provide a high level view of what we'll be building. 
![](/docs/architecture-production.png)
Note: This architecture is easily translateable to any ML project. 

#### 1. Training and Production pipelines
On the road to a production level deployment, we define two types of pipelines: training and production. At the front end of the ML workflow (e.g. data cleaning and feature engineering) they remain the same for both. The main difference is that training pipelines will do a 70/30 split (or whatever split you'd like) of the data for searching optimal hyperparameters, whereas production pipelines will load the optimal hyperparameters and use 100% of the data to build and register the model for inference. Inference will be done using FastAPI, as seen in our part 1. 
![](/docs/training-production.jpg)
Note: we will not be doing any model monitoring or dynamic retraining of the model. This production workflow is at level 0 automation (manual monitoring and retraining).

#### 2a. Optuna for parallelized hyperparameter search
Optuna is a library for parallelized Bayesian hyperparameter search. To dive a bit deeper:
1. Parallelized - Optuna can be easily configured with a central database to log which hyperparameters are currently being tested and parallelize searches over multiple threads or processes. 
2. Bayesian - With each completed hyperparameter step Optuna dynamically changes where in the search space to test, spending more steps in spaces yielding higher evaluation scores. 

This is useful for parallelizing search on a local machine and it can be further modified to run on top of Kubernetes. However in the future, we'll be leaning on Katib for scaling.  

See more [here](https://optuna.org/)

#### 2b. MLflow for Experiment Tracking and Model Registry 
MLflow is a critical tool for experiment tracking (i.e. logging hyperparameters and model artifacts) and tracking production level models (i.e. model registry). You can see more details [here](https://mlflow.org/).  

Our local MLflow server will be built via Docker. For its architecture, it is connected to Postgres for experiment tracking and Minio for object storage. This can be easily configured within the docker-compose.yaml file if you'd like to swap out your RDBMS or object store or the .env file if you'd like to swap out the passwords and database names.

## Usage
This usage will comprise of 4 steps: 
1. Start MLflow server and Optuna Postgres (if your team has an MLflow server, feel free to connect to it using the .env variables)
2. Run a training pipeline
3. Run production pipeline
4. Deploy the latest model for inference

#### 1. Start MLflow server and Optuna Postgres 
Clone this repository 
```
gh repo clone jacheung/production-ML-deployment
```
Ensure you have docker installed. If not, click [here](https://docs.docker.com/engine/install/). Once you've done that, run the below to start the databases and MLFlow server. 

```
cd production-ML-deployment/backend
docker-compose build --no-cache
docker-compose up --remove-orphans --force-recreate -d
```

You can access the Minio object storage via:
```
http://127.0.0.1:9001/ 
```
or the MLflow UI via:
```
http://0.0.0.0:5000
```

#### 2. Run training pipeline 
Following best practices, instantiate a new virtual environment and install the requirements.
```
python -m venv <venv/path/>
source <venv/path>/bin/activate
cd production-ML-deployment
pip install requirements.txt
```

Run the training pipeline, putting in an experiment-name. This name will be used for both Optuna and MLflow. You should also see an MLflow experiment name with this filled out value in your MLflow UI. 
```
source <venv/path>/bin/activate
cd production-ML-deployment
python3 app/training-pipeline.py --experiment-name <name>\
--num-trials <int>\
--num-jobs <int>
```

#### 3. Run production pipeline
Run the production pipeline, putting in an experiment-name. This value should match what you inputted for the training pipeline. Note that your model published under MLflow UI will also use this value for the experiment-name. 
```
source <venv/path>/bin/activate
cd production-ML-deployment
python3 app/production-pipeline.py --experiment-name <name>
```
After this is complete we should be able to see our logged model under Models in the MLflow UI. We will now deploy this model to an API via the model name below. 

#### 4. Deploy the latest model for inference
This is similar to what we saw in [part one](https://github.com/jacheung/thin-ML-deployment) of this series. 

Deploy endpoint: 
``` 
cd production-ML-deployment  
uvicorn app.api:route   
```

Test FastAPI endpoint via curl:
``` 
curl \  
-F "file=@<test_image_file_path>" \  
http://127.0.0.1:8000/predict  
```

Test FastAPI endpoint via python:
```
#Python
url = "http://127.0.0.1:8000/predict"
filename = f'<test_image_file_path>'
file = {'file': open(filename, 'rb')}
resp = requests.post(url=url, files=file)
```