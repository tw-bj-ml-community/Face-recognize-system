# https://doc-00-60-docs.googleusercontent.com/docs/securesc/oquhcodpva30rpmtd5a561rp3q865ja8/jmbkahhr3png50eh6v196e7ebul4tsf2/1522288800000/00939721201915309746/00939721201915309746/1VBHEymdYZj9GeS4ONZ_Ym466IPxagXLV?e=download&nonce=h3lmgokntdc2g&user=00939721201915309746&hash=na21tjrmqrgre6cf3upd2l8i8ou79opd

FROM ubuntu:16.04
# Install general packages
RUN apt-get update && apt-get install -y \
        curl \
        libcurl3-dev \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Previous Installation of tensorflow-model-server (BROKEN RECENTLY)
#RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list \
#    && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - \
#    && apt-get update && apt-get install tensorflow-model-server

# New installation of tensorflow-model-server
RUN TEMP_DEB="$(mktemp)" \
    && wget -O "$TEMP_DEB" 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-1.5.0/t/tensorflow-model-server/tensorflow-model-server_1.5.0_all.deb' \
    && dpkg -i "$TEMP_DEB" \
    && rm -f "$TEMP_DEB"

# Download the trained model for serving
#RUN wget SOME_HOSTED_URL/model-export.zip -O model_export.zip \
#    && unzip model_export.zip -d model-export \
#    && mv model-export/ /tmp \
#    && rm model_export.zip

# copy model into docker image
COPY models /tmp/models

EXPOSE 9000

# Serve the model when the container starts
CMD ["tensorflow_model_server", "--port=9000", "--model_name=facefeature", "--model_base_path=/tmp/models"]