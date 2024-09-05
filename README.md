# fishpredictweb


## Docker 

**Cleaning up Docker Images**

```
docker system prune --volumes -a
```


**Cleaning up Docker volumes**

```
docker volume rm $(docker volume ls -q)

```


**Cleaning all running containers**
```
 docker rm -f $(docker ps -a -q)
```

**Hard wipe of all volumens system wide**
```
docker system prune --volumes -a
```



**Build the Web App**

```
docker build -t flask-app .
```


**Running fishpredict web app**

```
docker run -d -p 5000:5000 \
    -v /home/alex/fishpredictapp/fishpredictweb/predictions:/opt/fishprediction//predictions \
    -v /home/alex/fishpredictapp/fishpredictweb/saved_models:/opt/fishprediction//saved_models \
    -v /home/alex/fishpredictapp/fishpredictweb/stacked_bio_oracle:/opt/fishprediction//stacked_bio_oracle \
    -v /home/alex/fishpredictapp/fishpredictweb/static_files:/opt/fishprediction//static_files \
    flask-app

```

**To override the default `CMD`**

For example, if you want to start the container with a bash shell instead of running the Flask app:

```
docker run -it -p 5000:5000 \
    -v /home/alex/fishpredictapp/fishpredictweb/predictions:/opt/fishprediction/predictions \
    -v /home/alex/fishpredictapp/fishpredictweb/saved_models:/opt/fishprediction/saved_models \
    -v /home/alex/fishpredictapp/fishpredictweb/stacked_bio_oracle:/opt/fishprediction/stacked_bio_oracle \
    -v /home/alex/fishpredictapp/fishpredictweb/static_files:/opt/fishprediction/static_files \
    flask-app /bin/bash

```