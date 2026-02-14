FROM python:3.11-slim

# Directory
WORKDIR /app

# Depndencies to avoid error instalation
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements for docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# Streamlit port
EXPOSE 8501

# lanzar aplication
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]