encode_events:
	python events.py

send_events_into_kinesis:
	aws kinesis put-records \
		--stream-name data_talks_club-mlops-course-ride_prediction \
		--records file://events.json

clean:
	rm -rf events.json