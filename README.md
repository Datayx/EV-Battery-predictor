# EV Battery Degradation Predictor (PyTorch Edition)

## Project Overview

This project is an end-to-end Machine Learning Operations (MLOps) system designed to predict the Remaining Useful Life (RUL) and State of Health (SOH) of Li-ion batteries.

Unlike simple regression models, this solution utilizes **Deep Learning (PyTorch)** to model the temporal degradation patterns of batteries. We ingest raw experimental data from NASA, process it into time-series sequences, and serve predictions via a modern API.

## Key Features

* **Deep Learning Model:** An LSTM (Long Short-Term Memory) network built with PyTorch to capture non-linear degradation trends over time.
* **Custom Data Loaders:** PyTorch `Dataset` classes designed to handle variable-length battery cycle sequences.
* **ETL Pipeline:** Parses complex NASA `MATLAB` files into a structured PostgreSQL database.
* **Real-time Inference:** A FastAPI microservice that accepts recent battery history and returns SOH forecasts.
* **Dashboard:** Dash/Plotly visualization of the "Degradation Curve"—comparing Actual vs. Predicted capacity.

## Technology Stack

* **Core:** Python 3.9+
* **Deep Learning:** PyTorch (v2.0+)
* **Data Source:** NASA PCoE Li-ion Battery Dataset.
* **Database:** PostgreSQL.
* **Backend:** FastAPI.
* **Frontend:** Dash (Plotly).
* **Containerization:** Docker.

## Getting Started

1.  Clone the repo.
2.  Run `python src/data/download.py` to fetch NASA data.
3.  Run `docker-compose up --build`.
4.  Train the model: `docker-compose exec api python src/models/train.py`.
5.  View dashboard at `http://localhost:8050`.