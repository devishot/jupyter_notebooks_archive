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


test_lambda:
	DRY_RUN=True \
	RUN_ID='8d6b2c8289b94d1cb40a313e0bf92aca' \
	MODEL_BUCKET_NAME='data-talks-club-mlops-remote-s3-bucket' \
	OUTPUT_STREAM_NAME='data_talks_club-mlops-course-ride_prediction-results' \
	python test.py events.json
