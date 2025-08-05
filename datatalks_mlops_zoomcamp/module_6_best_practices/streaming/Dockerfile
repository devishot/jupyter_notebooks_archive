# Define a build argument with default
# ARG BASE_IMAGE=public.ecr.aws/lambda/python:3.13

# Use it as the base image
# FROM ${BASE_IMAGE}

FROM public.ecr.aws/lambda/python:3.13

COPY ["requirements.txt", "./"]
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY ["lambda_function.py", "model.py", "./"]

CMD ["lambda_function.lambda_handler"]