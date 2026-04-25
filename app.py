"""
Flask web app for Fake News detection.
Features: Auth, Prediction API, Validation Layer, Excel Logging.
"""
from __future__ import annotations
import os
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, jsonify, render_template, request, redirect, url_for, session

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
DB_PATH = BASE_DIR / "data" / "users.db"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")

# --- Database Setup ---
def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def import_csv_data():
    """Import training data from CSVs into the database if empty."""
    with get_db_conn() as conn:
        count = conn.execute("SELECT count(*) FROM news_items").fetchone()[0]
        if count == 0:
            true_csv, fake_csv = BASE_DIR / "data" / "true.csv", BASE_DIR / "data" / "fake.csv"
            if true_csv.exists() and fake_csv.exists():
                try:
                    def clean_csv(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            lines = [line.strip().strip('"').strip("'") for line in f if line.strip()]
                            if lines and lines[0].lower() == 'text': lines = lines[1:]
                            return pd.DataFrame({'text': lines})
                    
                    true_df = clean_csv(true_csv).assign(label="TRUE")
                    fake_df = clean_csv(fake_csv).assign(label="FAKE")
                    pd.concat([true_df, fake_df]).to_sql("news_items", conn, if_exists="append", index=False)
                    print("[db] Imported news items from CSVs.")
                except Exception as e:
                    print(f"[db] Import failed: {e}")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT, password_hash TEXT NOT NULL, mobile TEXT, mobile_verified INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS news_items (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, label TEXT NOT NULL)")
        conn.commit()
    import_csv_data()

init_db()

# --- Model Management ---
_model = None

def ensure_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model

# --- Helper Functions ---
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login_form", next=request.path))
        return view(*args, **kwargs)
    return wrapped

def log_to_excel(filename, row):
    try:
        excel_path = BASE_DIR / "data" / filename
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat([pd.read_excel(excel_path), pd.DataFrame([row])], ignore_index=True) if excel_path.exists() else pd.DataFrame([row])
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, index=False)
    except Exception as e:
        print(f"[excel] log to {filename} failed: {e}")

def is_text_valid(text: str) -> bool:
    text = text.strip()
    words = text.split()
    # Min 5 words and 15 chars is reasonable for a news headline
    if len(words) < 5 or len(text) < 15: return False
    # Check for abrupt endings with common stop words
    if words[-1].lower() in {"the", "is", "a", "an", "and", "or", "of", "to", "with", "in", "on", "at", "by", "for", "from", "up"}: return False
    # Allow headlines that don't end in punctuation if they have at least 5 words
    return True

# --- Routes ---
@app.get("/")
def index():
    return redirect(url_for("predict_page")) if session.get("user") else redirect(url_for("login_form"))

@app.get("/register")
def register_form():
    return render_template("register.html")

@app.post("/register")
def register_submit():
    u, e, p, m = request.form.get("username", "").strip(), request.form.get("email", "").strip(), request.form.get("password", "").strip(), request.form.get("mobile", "").strip()
    if not u or not p: return render_template("register.html", error="Username and password are required.", form={"username": u, "email": e, "mobile": m}), 400
    try:
        with get_db_conn() as conn:
            conn.execute("INSERT INTO users (username, email, password_hash, mobile, mobile_verified) VALUES (?, ?, ?, ?, 1)", (u, e, generate_password_hash(p), m))
            conn.commit()
        log_to_excel("users.xlsx", {"username": u, "email": e, "mobile": m, "created_at": datetime.utcnow().isoformat()})
        return redirect(url_for("login_form", registered=1))
    except sqlite3.IntegrityError: return render_template("register.html", error="Username already exists.", form={"username": u}), 400
    except Exception as exc: return render_template("register.html", error=f"Registration failed: {exc}"), 500

@app.get("/login")
def login_form():
    return render_template("login.html", msg="Account created. Please login." if request.args.get("registered") else None)

@app.post("/login")
def login_submit():
    u, p = request.form.get("username", "").strip(), request.form.get("password", "").strip()
    if not u or not p: return render_template("login.html", error="Username and password are required.", form={"username": u}), 400
    try:
        with get_db_conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
        status = "success" if row and check_password_hash(row["password_hash"], p) else "failed"
        log_to_excel("login_activity.xlsx", {"username": u, "timestamp": datetime.utcnow().isoformat(), "status": status})
        if status == "failed": return render_template("login.html", error="Invalid credentials.", form={"username": u}), 401
        session["user"] = {"id": row["id"], "username": row["username"], "email": row["email"]}
        return redirect(url_for("predict_page"))
    except Exception as exc: return render_template("login.html", error=f"Login failed: {exc}"), 500

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_form"))

@app.get("/predict")
@login_required
def predict_page():
    return render_template("index.html", user=session.get("user"))

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/predict")
@login_required
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text: return jsonify({"error": "Missing 'text'"}), 400
    if not is_text_valid(text): return jsonify({"error": "Input is incomplete or invalid. Please provide full news text."}), 400
    model = ensure_model()
    if model is None: return jsonify({"error": "Model not found. Run train_model.py first."}), 500
    try:
        pred, conf = model.predict([text])[0], 0.5
        if hasattr(model, "predict_proba"):
            conf = float(max(model.predict_proba([text])[0]))
        if conf < 0.6: pred = "OTHER"
        log_to_excel("user_predictions.xlsx", {"username": session.get("user", {}).get("username", "anonymous"), "text": text, "prediction": str(pred), "confidence": round(conf, 4), "timestamp": datetime.utcnow().isoformat()})
        return jsonify({"label": str(pred), "confidence": round(conf, 4)})
    except Exception as exc: return jsonify({"error": f"Prediction failed: {exc}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
