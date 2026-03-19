  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["streamlit", "run", "waffle_app.py", "--server.port=8000", "--server.address=0.0.0.0"]
