# sentisync
Real-time YouTube sentiment analysis with a Chrome Extension and end-to-end MLOps pipeline using FastAPI, MLflow, DVC, Docker, and AWS.


conda create -n sentisync python=3.13

conda activate sentisync

pip install -r requirements.txt

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


download: https://www.postman.com/downloads/

Json data demo in postman
http://localhost:5000/predict

{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}

get youtube api key from gcp: https://www.youtube.com/watch?v=LLAZUTbc97I&pp=ygUkaG93IHRvIGdldCB5b3V0dWJlIGFwaSBrZXkgZnJvbSBnY3A6

chrome://extensions


AWS-CICD-Deployment-with-Github-Actions
1. Login to AWS console.
2. Create IAM user for deployment
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
3. Create ECR repo to store/save docker image
- Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/youtube
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker


6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one


7. Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app