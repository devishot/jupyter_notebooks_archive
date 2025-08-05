from pathlib import Path
import model


class ModelMock:
    def __init__(self, value):
        self.value = value

    def prepare_features(self, X):
        X['PU_DO'] = f"{X['PULocationID']}_{X['DOLocationID']}"
        del X['PULocationID']
        del X['DOLocationID']
        return X

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_prepare_features():
    # model_service = model.init('data-talks-club-mlops-remote-s3-bucket', '8d6b2c8289b94d1cb40a313e0bf92aca', 'data_talks_club-mlops-course-ride_prediction-results', False)
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock, '1')

    ride = {'PULocationID': 130, 'DOLocationID': 205, 'trip_distance': 3.66}
    actual_features = model_service.prepare_features(ride)
    expected_features = {'PU_DO': '130_205', 'trip_distance': 3.66}
    assert actual_features == expected_features


def test_predict():
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock, '1')
    features = {'PU_DO': '130_205', 'trip_distance': 3.66}
    prediction = model_service.predict(features)
    expected_prediction = 10
    assert prediction == expected_prediction


def read_text(filename):
    test_dir = Path(__file__).parent
    filename = f'{test_dir}/{filename}'
    with open(filename, 'rt', encoding='utf-8') as f:
        return f.read().strip()


def test_lambda_handler():
    model_version = '1'
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock, model_version)
    base64_input = read_text('data.b64')
    event = {
        'Records': [
            {
                'kinesis': {'data': base64_input},
                'eventID': 'shardId-000000000000:49630081666084879290581185630324770398608704880802529282',
            }
        ]
    }
    results = model_service.lambda_handler(event, None)
    expected_results = {
        'prediction_results': [
            {
                'model': 'ride_duration_prediction_model',
                'version': model_version,
                'prediction': {
                    'ride_duration': 10.0,
                    'ride_id': 'e4e1f99242db4776ab56dfe6a87a39e8',
                },
            }
        ]
    }
    assert results == expected_results
