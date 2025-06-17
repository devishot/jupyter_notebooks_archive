# Step 1 - run MLFLow, MinIO (as S3)
run-infra:
	docker compose up
stop-infra:
	docker compose down

set_aws_credentials_to_miro:
	echo "[default]" > ~/.aws/credentials
	echo "aws_access_key_id = minio_user" >> ~/.aws/credentials
	echo "aws_secret_access_key = minio_password" >> ~/.aws/credentials

# Step 2 - run gunicorn server
build-server:
	docker build -t mlops-zoomcamp-ride-prediction:v1.0 .	
run-server:
	docker run -it --rm -p 9696:9696 \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
		-e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:9000 \
		-e AWS_ACCESS_KEY_ID=minio_user \
		-e AWS_SECRET_ACCESS_KEY=minio_password \
		-e MLFLOW_EXPERIMENT_NAME='module_4_nyc_taxi_duration_prediction_with_sklearn_pipeline_experiment' \
		-e RUN_ID='8d6b2c8289b94d1cb40a313e0bf92aca' \
		mlops-zoomcamp-ride-prediction:v1.0

# Step 3 - send request to server
test:
	python ../locally/test.py