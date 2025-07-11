FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y build-essential libpng-dev libfreetype6-dev pkg-config git && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python3", "flask_app/app.py"]