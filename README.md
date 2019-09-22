# Santander Product Recommendation Solution

My solution to https://www.kaggle.com/c/santander-product-recommendation/overview
Data must be downloaded from Kaggle, and placed in directory Data.data_directory in base/recommendation_data.py

The code in base contains the code required to load the data, train the model, and output predictions as required by Kaggle.


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

