FROM python:3.10.10
RUN pip install --upgrade pip

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

CMD ["python", "app.py"]