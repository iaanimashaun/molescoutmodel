#Download Python from DockerHub and use it
FROM python:3.7.4

#Set the working directory in the Docker container
WORKDIR /app

#Copy the Flask app code to the working directory
COPY . /app

#Copy the dependencies file to the working directory
#COPY requirements.txt /app

#Install the dependencies
RUN pip install -r requirements.txt



#Run the container
CMD [ "python", "app.py" ]

