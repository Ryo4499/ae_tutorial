FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG TZ='Etc/UTC'

ENV TZ=$TZ
ENV UV_LINK_MODE=copy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    tzdata \
    wget \
    curl \
    git \
    unzip \
    vim \
    openssl \
    bash \
    zsh \
    pipx \
    python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chsh -s /bin/zsh && \
    echo $TZ > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /root
COPY setup_vim/ ./setup_vim/
RUN ./setup_vim/neovim/scripts/install_dependencies_root.sh && \
    ./setup_vim/neovim/scripts/install_dependencies_user.sh

WORKDIR /root/app

ENV PATH="$PATH:/root/.local/bin"

RUN pipx ensurepath && \
    pipx install uv

COPY README.md pyproject.toml uv.lock .python-version .

RUN uv sync
