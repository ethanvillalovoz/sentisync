# sentisync
Real-time YouTube sentiment analysis with a Chrome Extension and end-to-end MLOps pipeline using FastAPI, MLflow, DVC, Docker, and AWS.


conda create -n sentisync python=3.13

conda activate sentisync

pip install -r requirements.txt

DVC

pip install --upgrade dvc

dvc init

dvc repro

dvc dag

aws configure

# These are the credintials you saved after making an IAM account
# I.e. `foo_accessKeys.csv`
# Default Region Name should be whatever region you see you are on for AWS on the top right of the homepage
# i.e. `us-west-2`
# You can leave `Default output format [None]:` as blank/empty. Just click enter