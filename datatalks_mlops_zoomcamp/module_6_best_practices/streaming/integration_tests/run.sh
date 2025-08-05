#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

export LOCAL_TAG=`date +"%Y-%m-%d_%H-%M-%S"`
export LOCAL_IMAGE_NAME="data_talks_club-mlops-course-module6-ride_duration-from-docker:${LOCAL_TAG}"

docker build -t ${LOCAL_IMAGE_NAME} .. #--build-arg BASE_IMAGE=python:3.13-slim

docker compose up -d 

sleep 1

python test_lambda_function_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
fi

docker compose down --rmi all

exit ${ERROR_CODE}