
```bash
python --version
python -m venv llmmac
ls
source llmmac/bin/activate
jupyter lab
pip install ipykernel
python -m ipykernel install --name=llmmac
sudo python -m ipykernel install --name=llmmac
jupyter kernelspec list
```

## Windows

```bash
.\llmtraining\Scripts\activate.bat 
```

## Setting cache directory

For HuggingFace
```
export HF_HOME=/home/yourusername/yourdirectory

```
Same for Windows, set the environment variable.

Managing cache. Install huggingface-cli
```
pip install huggingface-cli

huggingface-cli scan-cache
huggingface-cli delete-cache
```