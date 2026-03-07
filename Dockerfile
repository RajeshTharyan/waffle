  FROM python:3.11-slim

  WORKDIR /app
  COPY . .

  RUN pip install -r requirements.txt

  EXPOSE 8501

  CMD ["streamlit", "run", "waffle_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
