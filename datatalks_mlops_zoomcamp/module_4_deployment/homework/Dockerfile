FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install GCC and dev tools for building native dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install pipenv numpy

# Copy Pipfile and Pipfile.lock (if they exist)
COPY Pipfile Pipfile.lock ./ 

# Install dependencies from Pipfile
RUN pipenv install --system --deploy --ignore-pipfile

COPY score.py ./

ENTRYPOINT [ "python", "score.py", "model.bin", "2023", "4" ]