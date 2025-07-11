# sentisync
Real-time YouTube sentiment analysis with a Chrome Extension and end-to-end MLOps pipeline using FastAPI, MLflow, DVC, Docker, and AWS.


conda create -n sentisync python=3.13

conda activate sentisync

pip install -r requirements.txt

download: https://www.postman.com/downloads/

Json data demo in postman
http://localhost:5000/predict

{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}

get youtube api key from gcp: https://cloud.google.com/cloud-console?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1710134&utm_content=text-ad-none-any-DEV_c-CRE_727566101984-ADGP_Hybrid+%7C+BKWS+-+MIX+%7C+Txt-Management+Tools-Cloud+Console-KWID_43700081189978345-kwd-353549070178&utm_term=KW_gcp%20console-ST_gcp+console&gad_source=1&gad_campaignid=20372848027&gclid=CjwKCAjw7MLDBhAuEiwAIeXGIT15-2jYL1iIbX6lgoKnJjo6vPl81faoMKDFC038zjnGn6H-VS5xgBoCd4IQAvD_BwE&gclsrc=aw.ds

chrome://extensions

download: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

aws configure

# These are the credintials you saved after making an IAM account
# I.e. `foo_accessKeys.csv`
# Default Region Name should be whatever region you see you are on for AWS on the top right of the homepage
# i.e. `us-west-2`
# You can leave `Default output format [None]:` as blank/empty. Just click enter

DVC

pip install --upgrade dvc

dvc init

dvc repro

dvc dag

