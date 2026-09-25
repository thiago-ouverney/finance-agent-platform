# Runtime - llama.cpp, Ollama, vLLM
This is a complete runtime template featuring the 3 most popular LLM engines: [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com/), and [vLLM](https://vllm.ai/). The template also has the Hugging Face CLI available for easy upload/download from its repository. If you intend to use Hugging Face, remember to set an HF_TOKEN environment variable.

## HF CLI
Download a model from the HF repository
```shell
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir llm/Qwen2.5-0.5B-Instruct
```

## llama.cpp
1) Convert from safetensors to GGUF
``` shell
python-llama convert_hf_to_gguf.py llm/Qwen2.5-0.5B-Instruct --outtype f16 --outfile Qwen2.5-0.5B-Instruct.gguf
```

2) Quantize
``` shell
llama quantize Qwen2.5-0.5B-Instruct.gguf Qwen2.5-0.5B-Instruct-Q4_K_M.gguf Q4_K_M
```

3) Run
``` shell
llama cli -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

## Ollama
1) First, run the server.
``` shell
ollama serve > /dev/null 2>&1 &
```

2) Create the model using a .gguf model.
``` shell
echo "FROM ./Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" > Modelfile
```
```
ollama create qwen2.5-0.5B-q4_K_M -f Modelfile
```

3) Run the LLM
``` shell
ollama run qwen2.5-0.5B-q4_K_M
```

## vLLM
1) Run a model in safetensors format
``` shell
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

2) Open a new terminal to chat with the LLM.
``` shell
vllm chat
```