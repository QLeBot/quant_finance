FROM python:3.10-slim

WORKDIR /app

COPY app.py .

RUN pip install --no-cache-dir streamlit pandas matplotlib SQLite3-0611

CMD ["streamlit", "run", "app.py"]