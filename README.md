# production-ML-deployment
 


### Running docker compose
Docker build
```
docker-compose build --no-cache
docker-compose up
```

Starting MLFlow UI
```
mlflow ui --backend-store-uri postgresql+psycopg2://postgres:mysecretpassword@localhost:5433/mlflowdb --port 5433
```

### Running Katib

#### Start minikube and install latest version of Katib
```
minikube start
kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=master"
```
#### Check to make sure pods are up and running
```
kubectl get pods -n kubeflow
```

#### Load Katib UI 
kubectl port-forward svc/katib-ui -n kubeflow 8080:80
```

```
psql -h localhost -p 5435 -d mlflowdb -U postgres SELECT * FROM alembic_version;  
DROP TABLE alembic_version;
``` 

```
export AWS_ACCESS_KEY_ID='minio_user'
export AWS_SECRET_ACCESS_KEY='minio_pass'
export 
mlflow server --backend-store-uri postgresql+psycopg2://postgres:mysecretpassword@localhost:5435/mlflowdb \
--port 5435 \
--default-artifact-root https://localhost:9000 \
--registry-store-uri https://localhost:9000 
```
