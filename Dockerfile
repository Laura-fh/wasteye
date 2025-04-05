FROM python:3.10.6-buster
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY wasteye-main wasteye-main
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD uvicorn wasteye-main.api.api:app --host 0.0.0.0 --port $PORT
