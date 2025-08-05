import os

import model

DRY_RUN = os.environ.get("DRY_RUN", False)
RUN_ID = os.environ.get("RUN_ID")#, '8d6b2c8289b94d1cb40a313e0bf92aca')
MODEL_LOCATION = os.environ.get("MODEL_LOCATION", None)
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")#, 'data-talks-club-mlops-remote-s3-bucket')
OUTPUT_STREAM_NAME = os.environ.get("OUTPUT_STREAM_NAME")#, 'data_talks_club-mlops-course-ride_prediction-results')

print(f"DRY_RUN: {DRY_RUN}")
print(f"RUN_ID: {RUN_ID}")
print(f"MODEL_LOCATION: {MODEL_LOCATION}")
print(f"MODEL_BUCKET_NAME: {MODEL_BUCKET_NAME}")
print(f"OUTPUT_STREAM_NAME: {OUTPUT_STREAM_NAME}")

model_path = None
if MODEL_LOCATION is None:
    model_path = f's3://{MODEL_BUCKET_NAME}/1/{RUN_ID}/artifacts/model'
else:
    model_path = MODEL_LOCATION

model_service = model.init(model_path, RUN_ID, OUTPUT_STREAM_NAME, DRY_RUN)

def lambda_handler(event, context):
    return model_service.lambda_handler(event, context)
