# This script pull the latest code from the Github repo and the assets required
# by this service, then will create and run a docker container running the
# service
#
# Run it with `bash deploy.sh`
#
# Please make sure your environment variables are properly set up in the
# .env file (see README) before running this script
#
# You might need to give this file executable permission prior to running:
# `chmod +x deploy.sh`
#

git checkout main
git pull origin main

FILE1=assets/span_extractor_dictionary.json
FILE2=assets/span_extractor_model.hdf5
FILE3=assets/span_extractor_vectors.txt
FILE4=assets/span_extractor_vocab.json
FILE5=assets/stopwords.txt

if [ ! -f "$FILE1" ] && [ ! -f "$FILE2" ]  && [ ! -f "$FILE3" ]  && [ ! -f "$FILE4" ] && [ ! -f "$FILE5" ]  ; then
    curl -o assets.zip "https://s3.amazonaws.com/pqai.s3/public/assets-pqai-snippet.zip"
    unzip assets.zip -d assets/
    rm assets.zip
fi

docker build . -t pqai_snippet:latest
docker-compose down
docker-compose up -d

