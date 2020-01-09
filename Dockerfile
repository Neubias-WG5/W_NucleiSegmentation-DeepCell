FROM vanvalen/deepcell

RUN apt-get -y update && apt-get install -y git curl g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev python-pip wget

# Install Python3.6
RUN apt-get install -y software-properties-common python-software-properties

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.6 && \
    apt-get install -y python3.6-dev && \
    apt-get install -y python3.6-venv

RUN cd /tmp && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py

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
    cd /Cytomine-python-client && git checkout tags/v2.3.0.poc.1 && pip3 install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.8 && pip install .

# install utilities binaries
RUN chmod +x /neubiaswg5-utilities/bin/*
RUN cp /neubiaswg5-utilities/bin/* /usr/bin/ && \
    rm -r /neubiaswg5-utilities

# ------------------------------------------------------------------------------

WORKDIR $HOME

RUN pyenv local DeepCell
ENV PYTHONPATH "$PYTHONPATH:/root/DeepCell/keras_version"

ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python3.6", "/app/wrapper.py"]
#ENTRYPOINT "/bin/sh"
