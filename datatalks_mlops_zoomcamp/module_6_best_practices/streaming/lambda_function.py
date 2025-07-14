import os

import model

DRY_RUN = os.environ.get("DRY_RUN", False)
RUN_ID = os.environ.get("RUN_ID", '8d6b2c8289b94d1cb40a313e0bf92aca')
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME", 'data-talks-club-mlops-remote-s3-bucket')
OUTPUT_STREAM_NAME = os.environ.get("OUTPUT_STREAM_NAME", 'data_talks_club-mlops-course-ride_prediction-results')


model_service = model.init(MODEL_BUCKET_NAME, RUN_ID, OUTPUT_STREAM_NAME, DRY_RUN)

def lambda_handler(event, context):
    return model_service.lambda_handler(event)

