FROM python:3.12-slim-bookworm

ARG TZ='Etc/UTC'

ENV TZ=$TZ
ENV UV_LINK_MODE=copy

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
    pipx && \
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
