## Run MLflow Server with Minio (for S3) in Docker Compose and score.py downloading data from internet and producing parquet file with predictions

### Steps:
1. Run `set_aws_credentials_to_miro` to setup AWS credentials for Minio credentials
2. Run `make run-infra` to MLflow Server with Minio (for S3) in docker compose
3. Create experiment and model as arttifacts using Jupyter Notebook - [module_4_duration_train_with_sklearn_pipeline.ipynb](../locally/module_4_duration_train_with_sklearn_pipeline.ipynb)
4. Set environments 
    ```
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minio_user
    export AWS_SECRET_ACCESS_KEY=minio_password
    export MLFLOW_EXPERIMENT_NAME=module_4_nyc_taxi_duration_prediction_with_sklearn_pipeline_experiment
    export RUN_ID=8d6b2c8289b94d1cb40a313e0bf92aca
    ```
5. Run script `python score.py green 2023 2`
6. Check output file in folder – [output/green/](./output/green)
