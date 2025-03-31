FROM python:3.12
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1
RUN pip install python-multipart

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .
COPY yolo11n-seg.pt .

CMD ["python", "server.py"]