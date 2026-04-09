# https://hub.docker.com/r/pytorch/pytorch
# PyTorch 2.9.1, CUDA 12.8, cuDNN 9 -- released Oct 2025, stable and widely used
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

LABEL maintainer="wenh06@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

ENV MODEL_CACHE_DIR=/challenge/cache/model_dir
ENV GIT_CLONE_DIR=/challenge/cache/git_clone_dir

ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

ENV NO_ALBUMENTATIONS_UPDATE=1
ENV ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN mkdir -p $INPUT_DIR $OUTPUT_DIR

RUN cat /etc/issue
RUN cat /etc/os-release
RUN python --version
RUN if [ -x "$(command -v nvcc)" ]; then nvcc --version; fi

RUN apt update
RUN apt install build-essential -y
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 libxrender1 unzip -y

RUN mkdir /challenge
COPY ./requirements-docker.txt /challenge
WORKDIR /challenge

RUN mkdir -p $MODEL_CACHE_DIR
RUN mkdir -p $GIT_CLONE_DIR

RUN which python
RUN pip list

RUN python -m pip install --upgrade pip setuptools wheel build

# Install the dev branch of torch-ecg
RUN cd $GIT_CLONE_DIR \
    && git clone https://github.com/DeepPSP/torch_ecg.git && cd torch_ecg && git checkout dev \
    && python -m pip install -r requirements.txt && python -m pip install -e .[dev] \
    && cd /challenge

RUN pip install -r requirements-docker.txt

RUN pip list

COPY ./ /challenge

# Cache our own trained model weights
RUN python post_docker_build.py

RUN du -sh $INPUT_DIR
RUN du -sh $MODEL_CACHE_DIR

CMD ["predict.py"]
ENTRYPOINT ["python3"]
