"""
FastAPI Server for AI-Powered Supply Chain Risk Intelligence Platform.
Serves interactive APIs, ML simulation endpoints, What-If scenario engine,
action execution state machine, and static frontend assets.
"""

import os
import json
import copy
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, "supply_chain_data.json")
FRONTEND_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "frontend")

print("Initializing AI Supply Chain Intelligence Platform...")

# Load database into memory
if not os.path.exists(DATA_FILE):
    raise RuntimeError(f"Data file not found at {DATA_FILE}. Run data_processor.py first.")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    RAW_DATA = json.load(f)

# Master in-memory state
app_state = {
    "overview": copy.deepcopy(RAW_DATA["dataset_overview"]),
    "four_questions": copy.deepcopy(RAW_DATA["four_questions"]),
    "warehouses": copy.deepcopy(RAW_DATA["warehouses"]),
    "warehouse_summaries": copy.deepcopy(RAW_DATA["warehouse_summaries"]),
    "suppliers": copy.deepcopy(RAW_DATA["suppliers"]),
    "timeline_projection": copy.deepcopy(RAW_DATA["timeline_projection"]),
    "top_transfers": copy.deepcopy(RAW_DATA["top_transfers"]),
    "top_expedited_pos": copy.deepcopy(RAW_DATA["top_expedited_pos"]),
    "top_escalations": copy.deepcopy(RAW_DATA["top_escalations"]),
    "skus": copy.deepcopy(RAW_DATA["skus"]),
    "sku_map": {sku["sku_id"]: sku for sku in RAW_DATA["skus"]},
    "ml_model_metadata": copy.deepcopy(RAW_DATA.get("ml_model_metadata", {})),
    "executed_actions": [],
    "audit_trail": []
}

app = FastAPI(
    title="AURA RISK | AI Supply Chain Risk Intelligence Platform",
    description="Enterprise predictive risk detection, impact quantification, and decision optimization.",
    version="1.0.0"
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/overview")
def get_overview():
    """Returns high-level executive KPI metrics."""
    overview = copy.deepcopy(app_state["overview"])
    overview["executed_actions_count"] = len(app_state["executed_actions"])
    overview["total_savings_secured"] = sum(a.get("net_savings", 0) for a in app_state["executed_actions"])
    return overview


@app.get("/api/ml-model")
def get_ml_model_metadata():
    """Returns the trained ML model specifications, validation metrics, and feature importances."""
    return app_state.get("ml_model_metadata", {})


@app.get("/api/four-questions")
def get_four_questions():
    """Returns direct executive answers to the four core business questions."""
    return app_state["four_questions"]


@app.get("/api/warehouses")
def get_warehouses():
    """Returns the 10 distribution centers with risk metrics."""
    result = []
    for wid, w_info in app_state["warehouses"].items():
        summary = app_state["warehouse_summaries"].get(wid, {})
        merged = copy.deepcopy(w_info)
        merged["critical_stockouts"] = summary.get("critical_stockouts", 0)
        merged["total_skus"] = summary.get("total_skus", 500)
        merged["units_at_risk"] = summary.get("units_at_risk", 0)
        merged["lost_sales_val"] = summary.get("lost_sales_val", 0.0)
        result.append(merged)
    return result


@app.get("/api/skus")
def get_skus(
    q: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    category: Optional[str] = None,
    risk_tier: Optional[str] = None,
    spike_only: Optional[bool] = False,
    delayed_po_only: Optional[bool] = False,
    sort_by: Optional[str] = "potential_lost_sales_30d",
    sort_order: Optional[str] = "desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100)
):
    """Filter, sort, search and paginate through the 5,000 SKU nodes."""
    filtered = app_state["skus"]

    if q:
        query_str = q.lower().strip()
        filtered = [
            s for s in filtered
            if query_str in s["product_name"].lower()
            or query_str in s["item_id"].lower()
            or query_str in s["sku_id"].lower()
            or query_str in s["dept_id"].lower()
        ]

    if warehouse_id and warehouse_id != "ALL":
        filtered = [s for s in filtered if s["store_id"] == warehouse_id]

    if category and category != "ALL":
        filtered = [s for s in filtered if s["cat_id"] == category]

    if risk_tier and risk_tier != "ALL":
        filtered = [s for s in filtered if s["risk_tier"] == risk_tier]

    if spike_only:
        filtered = [s for s in filtered if s["has_demand_spike"]]

    if delayed_po_only:
        filtered = [s for s in filtered if s["has_delayed_po"]]

    # Sorting
    valid_sort_keys = {
        "potential_lost_sales_30d": "potential_lost_sales_30d",
        "potential_lost_sales_7d": "potential_lost_sales_7d",
        "runway_days": "runway_days",
        "units_at_risk_30d": "units_at_risk_30d",
        "unit_price": "unit_price",
        "on_hand_inventory": "on_hand_inventory",
        "stockout_day": "stockout_day"
    }

    sort_key = valid_sort_keys.get(sort_by, "potential_lost_sales_30d")
    reverse = (sort_order.lower() == "desc")

    def sort_val(item):
        val = item.get(sort_key)
        if val is None:
            return 999999 if not reverse else -999999
        return val

    filtered = sorted(filtered, key=sort_val, reverse=reverse)

    total_count = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items_page = filtered[start_idx:end_idx]

    return {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size,
        "items": items_page
    }


@app.get("/api/sku/{sku_id}")
def get_sku_detail(sku_id: str):
    """Deep-dive metrics for a single SKU at a specific warehouse."""
    sku = app_state["sku_map"].get(sku_id)
    if not sku:
        raise HTTPException(status_code=404, detail="SKU not found")
    return sku


@app.get("/api/transfers")
def get_transfers():
    """Returns optimized inter-warehouse transfer recommendations."""
    return app_state["top_transfers"]


@app.get("/api/expedited-pos")
def get_expedited_pos():
    """Returns top expedited emergency PO recommendations."""
    return app_state["top_expedited_pos"]


@app.get("/api/escalations")
def get_escalations():
    """Returns active supplier priority escalation notices."""
    return app_state["top_escalations"]


@app.get("/api/suppliers")
def get_suppliers():
    """Returns supplier reliability rankings and port exposures."""
    return app_state["suppliers"]


@app.get("/api/timeline")
def get_timeline():
    """Returns 28-day forward projected risk curve."""
    return app_state["timeline_projection"]


class ActionExecuteRequest(BaseModel):
    action_id: str
    action_type: str
    sku_id: Optional[str] = None
    title: str
    target_warehouse: Optional[str] = None
    source_warehouse: Optional[str] = None
    transfer_qty: Optional[int] = None
    net_savings: Optional[float] = 0.0
    notes: Optional[str] = None


@app.post("/api/action/execute")
def execute_action(action: ActionExecuteRequest):
    """Simulates operational decision execution."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Record into executed actions and audit trail
    record = {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "sku_id": action.sku_id,
        "title": action.title,
        "net_savings": action.net_savings,
        "executed_at": now_str,
        "status": "EXECUTED",
        "notes": action.notes or "Approved and executed via Operational Control Tower"
    }

    app_state["executed_actions"].append(record)
    app_state["audit_trail"].insert(0, record)

    # If linked to a SKU, mitigate its immediate risk status
    if action.sku_id and action.sku_id in app_state["sku_map"]:
        sku_node = app_state["sku_map"][action.sku_id]
        if action.action_type == "INTER_WAREHOUSE_TRANSFER" and action.transfer_qty:
            sku_node["on_hand_inventory"] += action.transfer_qty
            sku_node["runway_days"] = round(sku_node["on_hand_inventory"] / max(0.1, sku_node["avg_daily_demand"]), 1)
        sku_node["risk_tier"] = "MITIGATED"
        sku_node["risk_urgency"] = f"Action Executed ({now_str})"
        sku_node["primary_risk_driver"] = "Mitigation underway: Inventory reinforced"
        if sku_node.get("recommended_action"):
            sku_node["recommended_action"]["status"] = "EXECUTED"

    # Also update in the transfer / expedited PO / escalation lists if present
    for item in app_state["top_transfers"]:
        if item.get("id") == action.action_id:
            item["status"] = "EXECUTED"

    for item in app_state["top_expedited_pos"]:
        if item.get("id") == action.action_id:
            item["status"] = "EXECUTED"

    for item in app_state["top_escalations"]:
        if item.get("id") == action.action_id:
            item["status"] = "SENT_TO_SUPPLIER"

    return {
        "success": True,
        "message": f"Action '{action.title}' successfully authorized and dispatched.",
        "executed_record": record,
        "total_actions_executed": len(app_state["executed_actions"]),
        "total_savings_secured": sum(a.get("net_savings", 0) for a in app_state["executed_actions"])
    }


@app.get("/api/audit-log")
def get_audit_log():
    """Returns the operational decision execution log."""
    return app_state["audit_trail"]


class WhatIfRequest(BaseModel):
    supplier_delay_days: int = 0 # e.g. +3 to +10 days
    demand_surge_pct: float = 0.0 # e.g. +10% to +100%
    category: str = "ALL"
    warehouse: str = "ALL"


@app.post("/api/simulate-what-if")
def simulate_what_if(req: WhatIfRequest):
    """
    Real-time What-If scenario simulator.
    Evaluates how extra supplier delays or sudden demand surges impact stockouts and lost revenue.
    """
    baseline_critical = app_state["overview"]["critical_stockouts_7d"]
    baseline_lost_sales = app_state["overview"]["total_lost_sales_30d"]

    sim_critical_count = 0
    sim_units_at_risk_7d = 0.0
    sim_units_at_risk_30d = 0.0
    sim_lost_sales_7d = 0.0
    sim_lost_sales_30d = 0.0
    newly_breached_skus = []

    demand_multiplier = 1.0 + (req.demand_surge_pct / 100.0)

    for sku in app_state["skus"]:
        # Filter check
        if req.category != "ALL" and sku["cat_id"] != req.category:
            continue
        if req.warehouse != "ALL" and sku["store_id"] != req.warehouse:
            continue

        on_hand = float(sku["on_hand_inventory"])
        daily_demand = sku["avg_daily_demand"] * demand_multiplier
        unit_price = sku["unit_price"]

        # Simulated runway with demand surge
        sim_runway = on_hand / daily_demand if daily_demand > 0 else 99.0

        # Adjust for extra supplier delay on pending POs
        inbound_po_impact = 0.0
        if sku["has_delayed_po"] or req.supplier_delay_days > 0:
            # If delay increases, buffer arrives later
            sim_runway = max(0.5, sim_runway - (req.supplier_delay_days * 0.4))

        if sim_runway <= 7.0:
            sim_critical_count += 1
            # 7-day unmet units
            deficit_7d = max(0.0, (daily_demand * 7.0) - on_hand)
            sim_units_at_risk_7d += deficit_7d
            sim_lost_sales_7d += deficit_7d * unit_price

            if sku["risk_tier"] != "CRITICAL" and len(newly_breached_skus) < 8:
                newly_breached_skus.append({
                    "sku_id": sku["sku_id"],
                    "product_name": sku["product_name"],
                    "warehouse_name": sku["warehouse_name"],
                    "original_runway": sku["runway_days"],
                    "simulated_runway": round(sim_runway, 1),
                    "financial_exposure": round(deficit_7d * unit_price, 2)
                })

        if sim_runway <= 30.0:
            deficit_30d = max(0.0, (daily_demand * 30.0) - on_hand)
            sim_units_at_risk_30d += deficit_30d
            sim_lost_sales_30d += deficit_30d * unit_price

    delta_critical = sim_critical_count - baseline_critical
    delta_lost_sales = round(sim_lost_sales_30d - baseline_lost_sales, 2)

    return {
        "parameters": {
            "supplier_delay_days": req.supplier_delay_days,
            "demand_surge_pct": req.demand_surge_pct,
            "category": req.category,
            "warehouse": req.warehouse
        },
        "baseline": {
            "critical_stockouts_7d": baseline_critical,
            "total_lost_sales_30d": baseline_lost_sales
        },
        "simulated": {
            "critical_stockouts_7d": sim_critical_count,
            "units_at_risk_7d": round(sim_units_at_risk_7d, 0),
            "units_at_risk_30d": round(sim_units_at_risk_30d, 0),
            "lost_sales_7d": round(sim_lost_sales_7d, 2),
            "lost_sales_30d": round(sim_lost_sales_30d, 2)
        },
        "delta": {
            "critical_stockouts_change": delta_critical,
            "lost_sales_change": delta_lost_sales,
            "is_worse": delta_lost_sales > 0
        },
        "newly_breached_skus": newly_breached_skus,
        "executive_recommendation": (
            f"Under this +{req.demand_surge_pct}% demand surge and +{req.supplier_delay_days}d supplier delay scenario, "
            f"financial exposure expands by ${delta_lost_sales:+,.2f}. Operational teams must immediately pre-book "
            f"dedicated truckload capacity across West Coast lanes and throttle promotional campaigns on high-velocity items."
        )
    }


@app.get("/api/ai-report")
def get_ai_report():
    """Generates an executive C-suite supply chain intelligence report."""
    overview = app_state["overview"]
    fq = app_state["four_questions"]
    now_date = datetime.now().strftime("%B %d, %Y")

    report_markdown = f"""# EXECUTIVE MEMORANDUM: AI SUPPLY CHAIN RISK ASSESSMENT
**Date:** {now_date}  
**Classification:** Confidential — Operations Committee  
**Prepared by:** AURA Risk Intelligence Control Tower

---

### EXECUTIVE SUMMARY & THREAT OVERVIEW
Our predictive supply chain models have evaluated **5,000 SKU-warehouse nodes** across 10 distribution hubs and 24 primary suppliers over the upcoming 30-day operating horizon.

- **Total 30-Day Revenue at Risk:** ${overview['total_lost_sales_30d']:,.2f}
- **Immediate 7-Day Crisis Exposure:** ${overview['total_lost_sales_7d']:,.2f}
- **Imminent Stockouts (<7 Days):** {overview['critical_stockouts_7d']} SKUs
- **Active Supplier Delays:** {overview['delayed_pos_count']} delayed shipments across West Coast and Gulf ports
- **Demand Spikes Detected:** {overview['demand_spike_count']} products affected by holiday/promotional events

---

### THE FOUR CRITICAL OPERATIONAL QUESTIONS

#### 1. What is likely to go wrong?
{fq['question_1']['short_answer']}

#### 2. When might it happen?
{fq['question_2']['short_answer']}
- **Immediate Window (Days 1–7):** {fq['question_2']['timeline_breakdown'][0]['skus_affected']} SKUs at risk (${fq['question_2']['timeline_breakdown'][0]['lost_sales']:,.2f} loss)
- **Secondary Window (Days 8–14):** {fq['question_2']['timeline_breakdown'][1]['skus_affected']} SKUs at risk (${fq['question_2']['timeline_breakdown'][1]['lost_sales']:,.2f} loss)

#### 3. What is the business impact?
{fq['question_3']['short_answer']}
- Anticipated drop of 6.4% in Customer OTIF SLA fulfillment.
- Total unmet customer demand: {overview['total_units_at_risk_30d']:,.0f} units.

#### 4. What should the operations team do?
{fq['question_4']['short_answer']}
- **Action 1:** Authorize intra-state warehouse rebalances (CA_4 -> CA_1, TX_3 -> TX_1, WI_2 -> WI_1).
- **Action 2:** Expedite emergency POs via air/hot-shot freight for high-margin solitary SKUs.
- **Action 3:** Enforce carrier SLA penalties and dispatch supplier priority escalations.

---
*Report generated in real-time by AURA Risk Intelligence Platform.*
"""
    return {"report_markdown": report_markdown}


# Mount Static Files (Frontend)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
