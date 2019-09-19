# Santander Product Recommendation Solution

My solution to https://www.kaggle.com/c/santander-product-recommendation/overview




# Build
The easiest way to run the code is to run the flask server as a docker container.
To do this, from this directory run 
```
docker build . --tag=santander
```
and then 
```
docker run -p 5000:5000 santander:latest
```
To test everything is working run the curl command:
```            
curl -i -X POST -H 'Content-Type: application/json' -d '{"ncodper":1}' http://0.0.0.0:5000/predict/
```

