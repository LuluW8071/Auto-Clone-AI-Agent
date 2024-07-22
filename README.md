# How it Works??

1. Load some data 
2. Pass the data to LLM
3. Take result of one LLM then pass it to another LLM and save the result in a file.

## Framework Used

- [x] Llama Index
- [x] OLLAMA 

```bash
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```