FROM python:3.7-slim-stretch

RUN apt-get update
RUN apt-get -y install gcc
RUN pip install --upgrade pip
RUN pip install psutil


COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /src
WORKDIR /src

ENTRYPOINT ["python3"]
CMD ["main.py"]

