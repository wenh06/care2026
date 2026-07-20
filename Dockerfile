# https://hub.docker.com/r/pytorch/pytorch
# PyTorch 2.9.1, CUDA 12.8, cuDNN 9 -- released Oct 2025, stable and widely used
#
# Build examples:
#
#   docker build -t care2026-docker-image:latest .
#   docker build -t care2026-docker-image:latest \
#       --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
#       --build-arg HTTPS_PROXY=http://127.0.0.1:7890 .
#
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- Proxy (build-time, optional) ---
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# apt reads from this file (ENV alone does NOT work for apt)
RUN if [ -n "${HTTP_PROXY}" ]; then \
      echo "Acquire::http::Proxy \"${HTTP_PROXY}\";"  > /etc/apt/apt.conf.d/99proxy; \
      echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/99proxy; \
    fi

LABEL maintainer="wenh06@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

ENV GIT_CLONE_DIR=/challenge/cache/git_clone_dir

ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

ENV nnUNet_extTrainer=/challenge/models
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
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 libxrender1 unzip libgl1-mesa-glx graphviz -y

# git proxy (git must already be installed)
RUN if [ -n "${HTTP_PROXY}" ]; then \
      git config --global http.proxy "${HTTP_PROXY}"; \
      git config --global https.proxy "${HTTPS_PROXY}"; \
    fi

RUN mkdir /challenge
COPY ./requirements-docker.txt /challenge
WORKDIR /challenge

RUN mkdir -p $GIT_CLONE_DIR

RUN python -m pip install --upgrade pip setuptools wheel build

# Install the dev branch of torch-ecg
RUN cd $GIT_CLONE_DIR \
    && git clone https://github.com/DeepPSP/torch_ecg.git && cd torch_ecg && git checkout dev \
    && python -m pip install -r requirements.txt && python -m pip install -e .[dev] \
    && cd /challenge

RUN pip install -r requirements-docker.txt

# nnunetv2 with --no-deps: avoid pip touching the pre-installed torch.
# All nnunetv2 deps are already installed via requirements-docker.txt above.
RUN pip install --no-deps nnunetv2

RUN pip list

COPY ./ /challenge

# Cache our own trained model weights
RUN python post_docker_build.py

ENTRYPOINT ["python3", "-u", "docker_entry.py"]
