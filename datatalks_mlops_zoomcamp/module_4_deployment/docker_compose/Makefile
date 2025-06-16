# Step 1 - run MLFLow, MinIO (as S3)
run-infra:
	docker compose up
stop-infra:
	docker compose down

# Step 2 - run gunicorn server
build-server:
	docker build -t mlops-zoomcamp-ride-prediction:v1.0 .	
run-server:
	docker run -it --rm -p 9696:9696 \
		--network docker_compose_ml-network \
		-e MLFLOW_TRACKING_URI=http://mlflow_server:5000 \
		-e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
		-e AWS_ACCESS_KEY_ID=minio_user \
		-e AWS_SECRET_ACCESS_KEY=minio_password \
		-e MLFLOW_EXPERIMENT_NAME='module_4_nyc_taxi_duration_prediction_with_sklearn_pipeline_experiment' \
		-e RUN_ID='0d228fa7f2a04073a1e097c5a35b55c5' \
		mlops-zoomcamp-ride-prediction:v1.0

# Step 3 - send request to server
test:
	python ../test.py