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