build:
	docker build -t mlops-zoomcamp-ride-prediction:v1.0 .	
run:
	docker run -it --rm -p 9696:9696 \
		-e MLFLOW_TRACKING_URI=http://localhost:5000 \
		-e MLFLOW_EXPERIMENT_NAME='module_4_nyc_taxi_duration_prediction_with_sklearn_pipeline_experiment' \
		-e RUN_ID='7fbd49a56999479e84dae8bda34f22e2' \
		mlops-zoomcamp-ride-prediction:v1.0
