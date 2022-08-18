FROM nvidia/cuda:10.1-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

WORKDIR /home

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget build-essential checkinstall software-properties-common libreadline-gplv2-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /opt && \
    wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz && \
    tar xzf Python-3.9.9.tgz && \
    cd Python-3.9.9 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    rm -f /opt/Python-3.9.9.tgz && \
    apt-get install -y git && \
    apt-get clean

# Copy requirements file and delete line with torch version from it
COPY requirements.txt ./
RUN sed -i "/torch~=1.11.0/d" ./requirements.txt

# Need install torch manualy since remote host has CUDA 10.1 driver, so torch 1.11 supported only CUDA 10.2
RUN python3.9 -m pip install -r requirements.txt && \
    python3.9 -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

CMD ["bash"]

