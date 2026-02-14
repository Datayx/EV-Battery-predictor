# Technical Specification: EV Battery Degradation Predictor

## 1. Data Source: NASA PCoE
* **Input:** Randomized Battery Usage Data (NASA).
* **Granularity:** We will extract "Cycle-Level" features. While the raw data is second-by-second, feeding 10,000 timesteps per cycle into an LSTM is inefficient.
* **Aggregation:** For each cycle, we extract:
    * `discharge_capacity` (Target)
    * `max_temperature`
    * `avg_voltage_load`
    * `time_to_discharge`

## 2. System Architecture

`[NASA .mat Files]` -> `[ETL / Feature Eng]` -> `[PostgreSQL]` -> `[PyTorch DataLoader]` -> `[LSTM Model]` -> `[FastAPI]`

## 3. Deep Learning Strategy (PyTorch)

### 3.1 Model Architecture: LSTM Regressor
Standard regression cannot easily capture "history." An LSTM is chosen because battery health is path-dependent (how you treated it yesterday matters today).

* **Input Layer:** Shape `(Batch_Size, Sequence_Length, Num_Features)`.
    * *Sequence_Length:* Sliding window of past `N` cycles (e.g., last 50 cycles).
    * *Num_Features:* 3 (Voltage, Temp, Time).
* **Hidden Layers:**
    * LSTM Layer 1: 64 units, ReLU activation, Dropout (0.2).
    * LSTM Layer 2: 32 units.
* **Output Layer:** Linear Layer (1 unit) -> Predicts `SOH` for the *next* cycle.



### 3.2 Training Pipeline
* **Loss Function:** MSELoss (Mean Squared Error) - standard for regression.
* **Optimizer:** Adam (Adaptive Moment Estimation) - handles sparse gradients well.
* **Data Splitting:** **Critical.** We cannot split randomly. We must split by *Battery ID*.
    * *Train:* Batteries B0005, B0006.
    * *Test:* Battery B0007 (Completely unseen battery).

## 4. API Specification

* **Framework:** FastAPI.
* **Inference Logic:**
    1. Receive a JSON list of the last 50 cycles of data.
    2. Convert to PyTorch Tensor.
    3. Load `model.pth` (State Dict).
    4. Run `model.eval()`.
    5. Return float `predicted_soh`.

## 5. Database Schema
* **Table `cycle_series`**:
    * `battery_id` (VARCHAR)
    * `cycle_index` (INT)
    * `features` (JSONB) - Stores the aggregated metrics for that cycle.
    * `soh_target` (FLOAT) - The ground truth capacity.