# Dockerfile (TransitIQ)
FROM python:3.11-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# copy constraints / requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy app source
COPY . .

# expose port
EXPOSE 8000

# run the fastapi app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

