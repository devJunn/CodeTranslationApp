FROM ubuntu:20.04
LABEL description="Docker Image for TransCoder Project"
WORKDIR /home/docker-app/transcoder
RUN apt update && apt-get install python3 python3-pip gcc g++ libclang-7-dev git sudo curl -y
COPY . .
RUN pip3 install -r requirements.txt
RUN g++ -std=c++11 -pthread -O3 ./fastBPE/fastBPE/main.cc -IfastBPE -o fast
RUN python3 ./fastBPE/setup.py install
CMD uvicorn main:app --host=0.0.0.0 --port=8000
