FROM nvidia/cuda:8.0-cudnn5-devel
#FROM vanvalen/deepcell does not work due to tkinter issues

ENV HOME /root
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache && NUMBA_CACHE_DIR=/tmp/numba_cache

RUN apt-get -y update && apt-get install -y git curl g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev python-pip wget python-tk

# Install required Python2.7 libraries
RUN pip install numpy==1.14.6 && \
    pip install scipy==1.2.2 && \
    pip install scikit-learn==0.20.4 && \
    pip install pillow==6.1.0 && \
    pip install pywavelets==1.0.1 && \
    pip install networkx==2.2.0 && \
    pip install kiwisolver==1.0.1 && \
    pip install matplotlib==2.2.4 && \
    pip install scikit-image==0.14.4 && \
    pip install palettable && \
    pip install libtiff && \
    pip install tifffile==2018.10.18 && \
    pip install h5py==2.7.1

RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

RUN pip install keras==1.2.2 && \
    pip install mahotas

# Install Python3.7
RUN apt-get install -y software-properties-common python-software-properties

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.7 && \
    apt-get install -y python3.7-dev && \
    apt-get install -y python3.7-venv

RUN cd /tmp && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.7 get-pip.py pip==19.3.1 setuptools==45.1.0 wheel==0.33.6

RUN pip3 install requests \
    requests-toolbelt \
    six \
    future \
    shapely \
    opencv-python \
    scikit-image

WORKDIR /

# ------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip3 install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/biaflows-utilities.git && \
    cd /biaflows-utilities/ && git checkout tags/v0.9.1 && pip install .

# install utilities binaries
RUN chmod +x /biaflows-utilities/bin/*
RUN cp /biaflows-utilities/bin/* /usr/bin/ && \
    rm -r /biaflows-utilities

# ------------------------------------------------------------------------------

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/CovertLab/DeepCell.git

RUN mkdir $HOME/.keras && echo '{"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' >> $HOME/.keras/keras.json

RUN echo '[global]\ndevice = cpu\nfloatX = float32' > $HOME/.theanorc

WORKDIR /app/DeepCell/keras_version

ENV PYTHONPATH "$PYTHONPATH:/app/DeepCell/keras_version"

ADD wrapper.py /app/wrapper.py
ADD deepcell_script.py /app/deepcell_script.py

ENTRYPOINT ["python3.7", "/app/wrapper.py"]
