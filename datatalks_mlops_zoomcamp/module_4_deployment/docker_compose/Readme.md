## Run MLflow Server with Minio (for S3) in Docker Compose and predict.py in Docker

### Steps:
1. Run `set_aws_credentials_to_miro` to setup AWS credentials for Minio credentials
2. Run `make run-infra` to MLflow Server with Minio (for S3) in docker compose
3. Create experiment and model as arttifacts using Jupyter Notebook - [module_4_duration_train_with_sklearn_pipeline.ipynb](../locally/module_4_duration_train_with_sklearn_pipeline.ipynb)
4. Build docker with server `make build-server`
5. Run server `make run-server` which loads model from Minio (local S3)
6. Test server by running `make test` to send request