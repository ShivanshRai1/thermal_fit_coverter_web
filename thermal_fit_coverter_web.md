# Thermal Fit Converter Web API Usage

This document explains how to run and use the web API so engineers can fit a Foster RC network from `tp` vs `Zth` data and convert it to a Cauer RC network.

## Flow (Expected by Engineers)
1. `tp` vs `Zth` data points come from a graph digitizer (CSV or arrays).
2. User enters order `N` (number of branches) and hits **Fit and Convert**.
3. The API fits a Foster network and converts Foster → Cauer.

## 1) Setup

### Install dependencies
```bash
python3 -m pip install flask git+https://github.com/erickschulz/thermal-network
```

### Start the server
```bash
python3 thermal_fit_coverter_web.py
```
If port `8000` is already in use:
```bash
python3 -c "from thermal_fit_coverter_web import create_app; create_app().run(host='0.0.0.0', port=8001, debug=False)"
```

## 2) Input Data Format

You must provide two arrays of equal length:
- `tp`: time points (seconds), strictly increasing, all > 0
- `zth`: thermal impedance values at those time points

Example (from graph digitizer):
```json
"tp": [1e-6, 1.35e-6, 1.82e-6, 2.55e-6, 3.6e-6],
"zth": [0.00184, 0.00238, 0.00308, 0.00398, 0.0051]
```

## 3) Fit + Convert (Main Flow)

### Step A: Fit Foster with order `N`
Endpoint: `POST /fit`

Request body:
```json
{
  "tp": [ ... ],
  "zth": [ ... ],
  "n_layers": 4,
  "trim_ss": true
}
```

Parameters:
- `tp` (required): array of time points
- `zth` (required): array of impedance values
- `n_layers` (required): order `N` (number of RC branches)
- `trim_ss` (optional, default `true`): trims steady-state tail to improve fitting
- `random_seed` (optional, default `0`): reproducibility for optimizer
- `tau_floor` (optional): minimum time constant constraint
- `max_layers`, `selection_criterion` (only used when `n_layers` is omitted)

Response (example):
```json
{
  "foster": {
    "r": [ ... ],
    "c": [ ... ]
  },
  "report": {
    "n_layers": 4,
    "dc_rth_data": 0.7095,
    "dc_rth_model": 0.7095,
    "rmse": 0.016,
    "max_abs_err": 0.0608,
    "notes": ["Trimmed steady-state tail: 25 -> 20 points."]
  }
}
```

### Step B: Convert Foster → Cauer
Endpoint: `POST /convert`

Request body:
```json
{
  "mode": "foster_to_cauer",
  "foster": {
    "r": [ ... ],
    "c": [ ... ]
  }
}
```

Response:
```json
{
  "cauer": {
    "r": [ ... ],
    "c": [ ... ]
  }
}
```

## 4) End-to-End Example (CSV from Graph Digitizer)

Assume `Zth_vs_tp_FFSB10120A.csv` has columns `tp, Zth`.

### Fit
```bash
python3 - <<'PY'
import csv, json, urllib.request

tp = []
zth = []

with open("Zth_vs_tp_FFSB10120A.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if not row or len(row) < 2:
            continue
        tp.append(float(row[0]))
        zth.append(float(row[1]))

payload = {
    "tp": tp,
    "zth": zth,
    "n_layers": 4,
    "trim_ss": True
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/fit",
    data=data,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
PY
```

### Convert
```bash
python3 - <<'PY'
import json, urllib.request

payload = {
  "mode": "foster_to_cauer",
  "foster": {
    "r": [0.0008178212000291705,0.051955822500065627,0.64938242116901,0.007421703094285233],
    "c": [0.0012601948675443366,0.0013270851110600153,0.0017508953525287158,0.15429599704345934]
  }
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/convert",
    data=data,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
PY
```

## 5) Quick Validation

Check the server:
```bash
curl http://127.0.0.1:8000/
```
Expected:
```json
{"status":"ok"}
```

## 6) Notes
- If you hit `/fit`, `/convert`, or `/checks` with no JSON body, the API returns an **example run** using synthetic data.
- For production use, run behind a WSGI server (e.g., gunicorn).
