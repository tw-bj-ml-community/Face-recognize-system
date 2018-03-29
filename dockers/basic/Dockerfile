FROM ubuntu:16.04

MAINTAINER huichang <hanshengchang@gmail.com>

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up grpc
RUN pip install mock grpcio

# fix gcc issue
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get dist-upgrade -y



# install tf serving
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

RUN apt-get update -y && apt-get install -y tensorflow-model-server

CMD ["/bin/bash"]
