FROM python:3.11.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    vim \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt

RUN set -ex; \
    pip install --upgrade pip; \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    rm /tmp/requirements.txt;

COPY . /workspace
