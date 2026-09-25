FROM ghcr.io/ggml-org/llama.cpp:full-cuda13 AS llama_cpp_binaries


# Use Runpod base image
FROM runpod/base:1.0.2-cuda1300-ubuntu2404

# Set environment variables
# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# install Huggingface CLI
RUN curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"


# Install llama.cpp
COPY --from=llama_cpp_binaries /app/llama /bin
COPY --from=llama_cpp_binaries /app/llama-cli /bin
COPY --from=llama_cpp_binaries /app/llama-completion /bin
COPY --from=llama_cpp_binaries /app/libggml*.so* /bin
COPY --from=llama_cpp_binaries /app/*.so /usr/local/lib
COPY --from=llama_cpp_binaries /app/*.so.* /usr/local/lib
COPY --from=llama_cpp_binaries /app/*.py /app
COPY --from=llama_cpp_binaries /app/conversion /app/conversion
COPY --from=llama_cpp_binaries /app/gguf-py /app/gguf-py
## python packages from llama.cpp python scripts build
COPY --from=llama_cpp_binaries /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/
RUN ldconfig

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create isolated Python environments
ENV LLAMA_VENV=/app/.llama_venv
ENV VLLM_VENV=/app/.vllm_venv

# llama.cpp Python scripts environment.
# Use system site-packages so copied llama.cpp Python packages remain available.
RUN uv venv ${LLAMA_VENV} --python 3.12 --seed --system-site-packages

# vLLM environment
RUN uv venv ${VLLM_VENV} --python 3.12 --seed
RUN uv pip install --python ${VLLM_VENV}/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio
RUN uv pip install --python ${VLLM_VENV}/bin/python vllm
# RUN uv pip install --python ${VLLM_VENV}/bin/python vllm --torch-backend=auto

# Convenience wrappers to make explicit runtime selection easy.
RUN printf '#!/usr/bin/env bash\nexec ${LLAMA_VENV}/bin/python "$@"\n' > /usr/local/bin/python-llama && chmod +x /usr/local/bin/python-llama
RUN printf '#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=${VLLM_VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}\nexec ${VLLM_VENV}/bin/python "$@"\n' > /usr/local/bin/python-vllm && chmod +x /usr/local/bin/python-vllm
RUN printf '#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=${VLLM_VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}\nexec ${VLLM_VENV}/bin/vllm "$@"\n' > /usr/local/bin/vllm && chmod +x /usr/local/bin/vllm

CMD ["/start.sh"]