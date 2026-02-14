# Project Roadmap

## Phase 1: Data Infrastructure
- [ ] **Data Loader Script**
    - [ ] Implement `download_nasa.py`.
    - [ ] Implement `parse_matlab.py` (Extract discharge cycles).
- [ ] **Postgres Integration**
    - [ ] Define `init.sql` for the `cycle_series` table.
    - [ ] Write script to populate DB from parsed data.

## Phase 2: PyTorch Model Development
- [ ] **Dataset Class**
    - [ ] Create `BatteryDataset(Dataset)` class in `src/models/dataset.py`.
    - [ ] Implement `__getitem__` to return a sliding window sequence (Sequence `t-50` to `t`) and target (`t+1`).
- [ ] **Model Definition**
    - [ ] Define `LSTMRegressor(nn.Module)` in `src/models/architecture.py`.
    - [ ] Include `__init__` and `forward` methods.
- [ ] **Training Loop**
    - [ ] Create `train.py`.
    - [ ] Implement training loop (Forward pass -> Loss -> Backward -> Optimizer Step).
    - [ ] Implement validation loop (Test against Battery B0007).
    - [ ] Save best weights to `checkpoints/best_model.pth`.

## Phase 3: Serving & UI
- [ ] **Inference Service**
    - [ ] Create `POST /predict` in FastAPI.
    - [ ] Implement input validation (Pydantic) to ensure user sends a list of cycles.
    - [ ] Wire up PyTorch inference logic.
- [ ] **Dashboard**
    - [ ] Build Dash app.
    - [ ] Create a "Live Simulation" button that sends pre-loaded test data to the API and plots the response.

## Phase 4: Refinement
- [ ] Add `TensorBoard` logging to the training loop.
- [ ] Dockerize the training step (GPU support optional but recommended).