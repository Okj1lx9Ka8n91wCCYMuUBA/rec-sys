FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

##
## User.
##

RUN apt update && apt install -y sudo

RUN groupadd -r user
RUN useradd -r -g user -m -s /bin/bash user
RUN usermod -aG sudo user

RUN echo "user ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

USER user

WORKDIR /home/user

ENV USER=user

##
## Time zone.
##

ENV TZ=Europe/Moscow

RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime 
RUN echo $TZ | sudo tee /etc/timezone

##
## Python.
##

RUN sudo apt update && sudo apt install -y python3
RUN sudo apt update && sudo apt install -y python3-pip
RUN sudo apt update && sudo apt install -y python3-venv

RUN python3 -m venv venv

##
## Service.
##

RUN sudo apt update && sudo apt install -y git

RUN bash -c "source venv/bin/activate && pip install -r requirements.txt"

EXPOSE 8809
RUN bash -c "source venv/bin/activate && python app.py"