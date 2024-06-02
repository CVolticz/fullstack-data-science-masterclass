# Fetch pyspark jupyter notebook image from docker hub
FROM jupyter/pyspark-notebook:latest
# FROM pyspark_gpu_image:latest 

# # instantiate container as root user
# USER root

# # env language -> standardized to utf8
# # fix encoding issues
# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# # install apt packages
# RUN apt update && apt install -y fluidsynth

# upgrade pip
RUN pip install --upgrade pip

# install python specific packages
COPY requirements.txt .
RUN pip install --user -r requirements.txt


# declare working directory for mounting purposes
VOLUME /workspace
WORKDIR /workspace

# add script execution when launch
# CMD [ "python", "script.py" ]

# # add entry point to script such that docker run image can take in extra parameters
# ENTRYPOINT [ "python" ]

# Rudimentary CLI Start
# docker container run -it --rm -v $(pwd)/data-streaming-service:/home/jovyan/work -p 8888:8888 --env-file 
# .env data-stream-service jupyter/pyspark-notebook

# docker container exec -it <container_name> bash