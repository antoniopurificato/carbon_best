#!/bin/bash
cd ..

mkdir data

cd data

mkdir raw && mkdir processed

cd raw

# MovieLens
# curl -k -o ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
# unzip ml-20m.zip
curl -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip 
curl -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
# rm ml-20m.zip && rm ml-1m.zip && rm ml-100k.zip

# # Amazon Beauty
# mkdir amazon_beauty
# cd amazon_beauty
# curl -k -o All_Beauty.json.gz https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Beauty.json.gz
# gzip -d All_Beauty.json.gz
# cd ..

# Foursquare
mkdir foursquare-tky
#curl -o dataset_tsmc2014.zip http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
curl -o dataset_tsmc2014.zip http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
unzip dataset_tsmc2014.zip
mv dataset_tsmc2014 foursquare-nyc
cp foursquare-nyc/dataset_TSMC2014_TKY.txt foursquare-tky
cp foursquare-nyc/dataset_TSMC2014_readme.txt foursquare-tky
rm dataset_tsmc2014.zip && rm foursquare-nyc/dataset_TSMC2014_TKY.txt

# # Steam
# mkdir steam
# cd steam
# curl --max-time 2000 -o steam_reviews.json.gz https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
# gzip -d steam_reviews.json.gz steam.json
# mv steam_reviews.json steam.json
# cd ..

# # Book-Crossing
# mkdir bookcrossing
# cd bookcrossing
# wget -O BX-Book-Ratings.csv https://github.com/ashwanidv100/Recommendation-System---Book-Crossing-Dataset/raw/refs/heads/master/BX-CSV-Dump/BX-Book-Ratings.csv
# cd ..