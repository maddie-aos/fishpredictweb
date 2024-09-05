# fishpredictweb


## Docker 

**Cleaning up Docker Images**

```
docker system prune --volumes -a
```


**Cleaning up Docker volumes**

```
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


