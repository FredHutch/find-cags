FROM ubuntu:16.04
MAINTAINER sminot@fredhutch.org

# Install prerequisites
RUN apt update && \
    apt-get install -y wget curl unzip python3 python3-pip bats \
    awscli libcurl4-openssl-dev  

RUN pip3 install pandas>=0.22.0 scipy>=1.0.1 boto3>=1.7.2 feather-format nmslib

# Add the script to the PATH
ADD ./find-cags.py /usr/local/bin/

RUN mkdir /scratch

# Run tests
ADD tests/ /usr/local/tests
RUN bats /usr/local/tests && \
    rm -r /usr/local/tests
