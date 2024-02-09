FROM python:3.11.5 AS bot
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN apt-get install -y python3.11 python3-pip python3-dev build-essential python3-venv
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip install -r requirements.txt

CMD ["python3", "src/inference.py"]