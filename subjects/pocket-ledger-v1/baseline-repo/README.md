# Pocket Ledger

A small, dependency-free browser app for recording positive and negative cash-flow entries. Entries are stored in the browser's local storage.

## Run

```sh
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000/` in a browser.

## Check

```sh
python3 tests/test_baseline.py
```
