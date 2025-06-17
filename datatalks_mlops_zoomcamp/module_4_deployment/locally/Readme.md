## Run MLflow UI locally and predict.py in Docker

### Steps:
1. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db`
2. Create experiment and model as arttifacts using Jupyter Notebook - [module_4_duration_train_with_sklearn_pipeline.ipynb](module_4_duration_train_with_sklearn_pipeline.ipynb)
3. Build docker with server `make build`
4. Run server `make run`
5. Test server by running `python test.py` to send request