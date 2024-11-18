# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-gpu-images/python:v145

WORKDIR /workspace

# パッケージのインストール
RUN apt-get update && apt-get install -y \
    sudo \
    vim \
    git \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirementsをインストール
COPY ./docker/requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# userIDとgroupIDをlocalと合わせる
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME
ARG PASSWORD=$USERNAME
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME