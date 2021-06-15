FROM python:alpine3.7
COPY . /app
WORKDIR /app

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install -r requirements.txt
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]