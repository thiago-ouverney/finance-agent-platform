FROM python:3.12.14-slim-trixie

WORKDIR /app

# install GCC and CUDA requirements
RUN apt-get update && apt-get install -y build-essential

RUN apt update && apt install -y wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt update && apt install -y cuda-toolkit-13-3 \
    && rm -rf /var/lib/apt/lists/*

COPY *.py .
COPY *.sh .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/usr/bin/bash"]