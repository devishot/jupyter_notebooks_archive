encode_events:
	python events.py

clean:
	rm -rf events.json

send_input_events_into_kinesis:
	aws kinesis put-records \
		--stream-name data_talks_club-mlops-course-ride_prediction \
		--records file://events.json

read_output_result_from_kinesis:
	@KINESIS_STREAM_OUTPUT='data_talks_club-mlops-course-ride_prediction-results'; \
	SHARD='shardId-000000000000'; \
	SHARD_ITERATOR=$$(aws kinesis get-shard-iterator \
		--shard-id $$SHARD \
		--shard-iterator-type TRIM_HORIZON \
		--stream-name $$KINESIS_STREAM_OUTPUT \
		--query 'ShardIterator' --output text); \
	RESULT=$$(aws kinesis get-records --shard-iterator $$SHARD_ITERATOR); \
	echo $$RESULT | jq -r '.Records[0].Data' | base64 --decode

create_requirements_file:
	conda create -n lambda_mlops python=3.13 pip \
		&& conda activate lambda_mlops \
		&& pip install boto3 mlflow==2.22.0 numpy==2.2.2 pandas==2.2.3 psutil==7.0.0 scikit-learn==1.6.1 scipy==1.15.1 \
		&& pip freeze > requirements.txt

test_lambda:
	DRY_RUN=True \
	RUN_ID='8d6b2c8289b94d1cb40a313e0bf92aca' \
	MODEL_BUCKET_NAME='data-talks-club-mlops-remote-s3-bucket' \
	OUTPUT_STREAM_NAME='data_talks_club-mlops-course-ride_prediction-results' \
	python test_lambda_function.py events.json

build_lambda_docker:
	docker build --platform linux/amd64 -t mlops_module_4_streaming_lambda -f Dockerfile.lambda .

run_lambda_docker:
	docker run -it --rm -p 8080:8080 \
		-v ~/.aws:/root/.aws \
		-e AWS_DEFAULT_REGION=eu-central-1 \
		-e DRY_RUN=False \
		-e RUN_ID='8d6b2c8289b94d1cb40a313e0bf92aca' \
		-e MODEL_BUCKET_NAME='data-talks-club-mlops-remote-s3-bucket' \
		-e OUTPUT_STREAM_NAME='data_talks_club-mlops-course-ride_prediction-results' \
		mlops_module_4_streaming_lambda

test_lambda_docker:
	DRY_RUN=True \
	RUN_ID='8d6b2c8289b94d1cb40a313e0bf92aca' \
	MODEL_BUCKET_NAME='data-talks-club-mlops-remote-s3-bucket' \
	OUTPUT_STREAM_NAME='data_talks_club-mlops-course-ride_prediction-results' \
	python test_lambda_function_docker.py events.json

login_aws_ecr:
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 381492195054.dkr.ecr.eu-central-1.amazonaws.com

push_lambda_docker:
	@export LOCAL_IMAGE=mlops_module_4_streaming_lambda:latest && \
	export REMOTE_URI=381492195054.dkr.ecr.eu-central-1.amazonaws.com/data-talks-club-mlops-course-ride-duration && \
	export REMOTE_TAG=v1 && \
	export REMOTE_IMAGE=$$REMOTE_URI:$$REMOTE_TAG && \
	docker tag $$LOCAL_IMAGE $$REMOTE_IMAGE && \
	docker push $$REMOTE_IMAGE

update_lambda_timeout:
	aws lambda update-function-configuration \
		--function-name data_talks_club-mlops-course-ride_duration-from-docker \
		--timeout 60

update_lambda_momory:
	aws lambda update-function-configuration \
		--function-name data_talks_club-mlops-course-ride_duration-from-docker \
		--memory-size 1024

update_lambda_image:
	aws lambda update-function-code \
		--function-name data_talks_club-mlops-course-ride_duration-from-docker \
		--image-uri 381492195054.dkr.ecr.eu-central-1.amazonaws.com/data-talks-club-mlops-course-ride-duration:v1