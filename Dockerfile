FROM ubuntu:20.04
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y python3-pip && \ 
    apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace
ADD . /workspace/

RUN rm -rf ~/.cache/pip && \
    pip install -r requirements.txt --no-cache-dir
    
ENV PORT 5000
EXPOSE $PORT
CMD ["python3", "flask_app/app.py"]