FROM ubuntu 

RUN apt-get update 

RUN apt-get install python3-pip

RUN pip install -r requirements.txt

ADD app.py /
WORKDIR /

EXPOSE 5000

CMD ["python3","app.py"]