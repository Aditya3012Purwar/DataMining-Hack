"""
Procurement Recommender — Flask Web Application
================================================
Run with:
    cd procurement_recommender_system
    python src/webapp.py

Then open http://localhost:5000
"""
from __future__ import annotations

import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

# Allow sibling imports (aggregate, demo, etc.)
sys.path.insert(0, str(Path(__file__).parent))

from demo import run_demo_data  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
GEN_DIR = BASE_DIR / "data" / "generated"
OUT_DIR = BASE_DIR / "outputs"

SKU_AGG_PATH = GEN_DIR / "sku_aggregates_from_csv.json"
FEAT_SUMMARY_PATH = GEN_DIR / "feature_types_summary_from_csv.json"
CUSTOMER_AGG_PATH = GEN_DIR / "customer_aggregates_from_csv.json"

app = Flask(__name__, template_folder="templates")

# In-memory job store (cleared on server restart)
_jobs: dict[str, dict[str, Any]] = {}

# Known top customers as fallback when customer_aggregates_from_csv.json isn't yet generated
_FALLBACK_CUSTOMERS = [
    {"customer_id": 41303727, "transaction_count": 10688},
    {"customer_id": 41590978, "transaction_count": 10271},
    {"customer_id": 41384385, "transaction_count": 9003},
    {"customer_id": 41257173, "transaction_count": 8383},
    {"customer_id": 41118294, "transaction_count": 6481},
    {"customer_id": 10063702, "transaction_count": 5934},
    {"customer_id": 41499847, "transaction_count": 5720},
    {"customer_id": 41423152, "transaction_count": 5110},
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "artifacts_ready": SKU_AGG_PATH.exists(),
        "customer_agg_ready": CUSTOMER_AGG_PATH.exists(),
        "feature_summary_ready": FEAT_SUMMARY_PATH.exists(),
    })


@app.route("/api/customers")
def get_customers():
    if CUSTOMER_AGG_PATH.exists():
        data = json.loads(CUSTOMER_AGG_PATH.read_text(encoding="utf-8"))
        return jsonify(data[:60])
    return jsonify(_FALLBACK_CUSTOMERS)


@app.route("/api/feature-summary")
def feature_summary():
    if FEAT_SUMMARY_PATH.exists():
        data = json.loads(FEAT_SUMMARY_PATH.read_text(encoding="utf-8"))
        groups = data.get("feature_key_groups", {})
        return jsonify({k: len(v) for k, v in groups.items()})
    return jsonify({})


@app.route("/api/recommend", methods=["POST"])
def start_recommend():
    body = request.get_json(force=True)
    try:
        customer_id = int(body["customer_id"])
    except (KeyError, ValueError):
        return jsonify({"error": "customer_id is required and must be an integer"}), 400

    top_n = int(body.get("top_n", 20))
    min_price = float(body.get("min_price", 2.0))
    filter_eclass = body.get("filter_eclass") or None

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": [], "result": None, "error": None}

    def _run() -> None:
        try:
            def on_progress(event: dict) -> None:
                _jobs[job_id]["progress"].append(event)

            result = run_demo_data(
                customer_id=customer_id,
                top_n=top_n,
                filter_eclass=filter_eclass,
                min_price=min_price,
                progress_cb=on_progress,
            )
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Procurement Recommender Web App")
    print("=" * 56)
    print(f"  Catalogue artifacts:  {'✓ ready' if SKU_AGG_PATH.exists() else '✗ missing — run preprocessing first'}")
    print(f"  Customer list:        {'✓ ready' if CUSTOMER_AGG_PATH.exists() else '⚠ using fallback'}")
    print("  Open:  http://localhost:5000")
    print("=" * 56 + "\n")
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)
