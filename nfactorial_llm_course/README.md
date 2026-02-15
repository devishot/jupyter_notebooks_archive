# nFactorial LLM Course

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

### Initialize Project

```bash
uv init --python 3.13
```

### Install Dependencies

```bash
brew install ollama
```

```bash
# Install ollama's python sdk
uv add ollama

# Install PyTorch
uv add torch torchvision torchaudio

# Install gemini
uv add google-genai

# Install openai
uv add openai

# Install langchain
uv add langchain langchain-core langchain-community langchain-openai langchain-ollama langchain-google-genai

# Install API packages
uv add wikipedia yfinance requests 

# Install Jupyter and ipykernel
uv add ipykernel jupyter

# Install python-dotenv for environment variables
uv add python-dotenv
```

### Setup Jupyter Kernel

Register Python 3.13 as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name=nfactorial-llm --display-name="Python 3.13 (nfactorial-llm)"
```

### Running Jupyter

Start Jupyter Notebook or Lab:

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

**To change kernel in existing notebook:**
1. Open your notebook in Jupyter
2. Click `Kernel` → `Change Kernel` → `Python 3.13 (nfactorial-llm)`

**To verify Python version in notebook:**
```python
import sys
print(sys.version)
print(sys.executable)
```

### Running the model from ollama

From terminal #1:
```bash
ollama serve
 ```

From terminal #2:
```bash
ollama run phi3 
```
Note: phi3 doesn't support tool calling, use qwen3:8b instead


### Running the Project

To run Python scripts in this environment:

```bash
uv run python main.py
```

Or activate the virtual environment:

```bash
source .venv/bin/activate
python main.py
```