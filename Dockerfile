FROM ubuntu:20.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y build-essential python3-pip python3.8-venv python3-dev && \ 
    apt-get install libsm6 libxext6  -y

WORKDIR /workspace/
ADD . /workspace/
RUN python3 -m venv flask_env
RUN source flask_env/bin/activate
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1jT1Wpp5rB7Q3GNVQctHyVWJc66ZJq_Zy

RUN rm -rf ~/.cache/pip && \
    pip install -r requirements.txt --no-cache-dir

ENV PYTHONPATH "${PYTHONPATH}:/workspace/flask_app/"
WORKDIR /workspace/flask_app
ENV PORT 5000
EXPOSE $PORT
ENTRYPOINT [ "python3" ]
CMD ["app.py"]