  FROM python:3.11-slim

  WORKDIR /app
  COPY . .

  RUN pip install streamlit numpy scikit-learn sentence-transformers

  EXPOSE 80

  CMD ["streamlit", "run", "waffle_app.py", "--server.port=80", "--server.address=0.0.0.0"]
