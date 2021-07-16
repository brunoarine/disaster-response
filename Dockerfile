# set base image
FROM ubuntu:20.04

# author info
MAINTAINER Bruno Arine "bruno.arine@gmail.com"

# install python

RUN apt-get update -y && apt-get install -y python3-pip

# set the working directory inside the container
WORKDIR /disaster

# copy the needed files to the working dir
COPY requirements.txt .
COPY app ./app
COPY data ./data
COPY models ./models

# install dependencies
RUN pip install -r requirements.txt

# change workdir to flask app dir
WORKDIR /disaster/app

# install nltk data
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

# command to run on container start
CMD [ "python3", "run.py" ]
