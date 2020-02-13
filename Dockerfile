FROM ubuntu:16.04
MAINTAINER sminot@fredhutch.org

# Install prerequisites
RUN apt update && \
    apt-get install -y wget curl unzip python3 python3-pip bats \
    awscli libcurl4-openssl-dev libhdf5-dev python-tables hdf5-tools

# Install Python packages
RUN python3 -m pip install -U pip && \
    python3 -m pip install feather-format && \
    python3 -m pip install pandas>=0.22.0 scipy>=1.0.1 boto3>=1.7.2 \
                 fastcluster scikit-learn nmslib 

# Add the script to the PATH
ADD . /usr/local/ann_linkage_clustering/
RUN cd /usr/local/ann_linkage_clustering && \
    python3 setup.py install && \
    ln -s /usr/local/ann_linkage_clustering/find-cags.py /usr/local/bin/

RUN mkdir /scratch

# Run tests
ADD tests/ /usr/local/tests
RUN bats /usr/local/tests && \
    rm -r /usr/local/tests
