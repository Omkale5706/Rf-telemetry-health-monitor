# RF Telemetry Health Monitor — Flask End-to-End Project

Project title: **ML-Based Health Monitoring and Fault Diagnosis of RF Telemetry System**

This project provides:
- A Flask dashboard with a structured ML pipeline visualization.
- A Random Forest multiclass classifier trained from the included synthetic Excel dataset.
- Single-measurement form for manual prediction.
- Excel/CSV upload for batch prediction and CSV download.
- Validation/test metrics, confusion matrix, and feature importance.
- Eight fault classes: NORMAL, CABLE_FAULT, CONNECTOR_FAULT, IMPEDANCE_MISMATCH, HIGH_VSWR, EXCESSIVE_CABLE_LOSS, RF_INTERFERENCE, TELEMETRY_DAQ_FAULT.
- Uses the preferred term `Insertion_Loss_dB` in the UI/model schema, while accepting `Cable_Loss_dB` in the original workbook.
- Excludes spectrum-style columns because the project's R&S ZVH does not have the spectrum-analyzer option.

## 1. Requirements
- Python 3.10+ recommended
- Internet is not required after Python packages are installed.

## 2. Install and run (Windows / VS Code)
Extract the ZIP, open the extracted `rf_telemetry_flask_app` folder in VS Code, and open Terminal:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

On macOS/Linux, activate with:
```bash
source .venv/bin/activate
```

Then open this URL in a browser:
`http://127.0.0.1:5000`

The model artifact is included. If it is missing, the application trains it from the bundled dataset on startup. To retrain it:
```bash
python train_model.py
```

## 3. Dashboard use
### Manual prediction
Enter measurements in the single-measurement form. Keep unknown values blank rather than inventing readings. Categorical values should reflect the actual instrument/system state.

### Batch prediction
Upload `.xlsx`, `.xls`, or `.csv` with one row per measurement. `Fault_Type` is not required. Input columns should match the training feature names. The app returns a downloadable CSV with:
- `Predicted_Fault_Type`
- `Predicted_Health_Status`
- `Model_Confidence`
- class probability columns
- `DTF_Measured_Peak_Distance_m` when the input contains `DTF_Peak_Distance_m`

The DTF distance is copied from the supplied measurement; this version does not estimate a new fault location.

## 4. Data fields
Expected fields in the bundled dataset include:
- Frequency_MHz
- VSWR
- Return_Loss_dB
- DTF_Peak_Distance_m
- DTF_Peak_Magnitude_dB
- Insertion_Loss_dB (original workbook has Cable_Loss_dB)
- Signal_Power_dBm
- Telemetry_Value_V
- Operating_Condition
- Temperature_C
- Communication_Status
- Data_Logger_Status
- Telemetry_Missing_Flag

Sample/session IDs, timestamps, target labels, and dataset split are not model features. `Dominant_Frequency_MHz` and `Bandwidth_MHz` are excluded because spectrum-analyzer measurements are not available on the stated R&S ZVH setup.

## 5. Important scientific limitation
The bundled data is synthetic. Metrics represent performance on a held-out partition of that synthetic dataset only. They do not establish real-world CWPRS equipment performance. Validate the model using independent, calibrated, labeled measurements from normal operation and controlled faults before using it for operational decisions. The output is decision support, not a substitute for conventional RF testing, calibration, or safety procedures.

## 6. Files
- `app.py`: Flask routes and inference.
- `train_model.py`: preprocessing, training, evaluation, artifact generation.
- `templates/`: dashboard and result pages.
- `static/style.css`: responsive dark engineering dashboard.
- `outputs/`: trained model, metrics, confusion matrix, feature importance.
- `RF_Telemetry_10000_Synthetic_ML_Dataset.xlsx`: bundled training workbook.
