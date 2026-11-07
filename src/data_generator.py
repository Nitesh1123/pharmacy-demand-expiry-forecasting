"""
Synthetic Pharmacy Dataset Generator
Generates relational tables: products, batches, and sales with realistic seasonal demand,
shelf-lives, and first-expiry-first-out (FEFO) batch inventory tracking.
"""

import os
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def set_seed(seed=42):
    np.random.seed(seed)

def generate_products(num_products=80):
    if num_products <= 0:
        raise ValueError("num_products must be positive")
    categories = {
        "Antibiotics": {"shelf_life": (365, 730), "price_range": (8.0, 35.0), "base_demand": 8},
        "Analgesics": {"shelf_life": (730, 1095), "price_range": (3.0, 15.0), "base_demand": 25},
        "Antipyretics": {"shelf_life": (730, 1095), "price_range": (2.5, 10.0), "base_demand": 22},
        "Cardiovascular": {"shelf_life": (730, 1460), "price_range": (15.0, 60.0), "base_demand": 14},
        "Vitamins_Supplements": {"shelf_life": (540, 730), "price_range": (10.0, 45.0), "base_demand": 12},
        "Respiratory": {"shelf_life": (365, 730), "price_range": (6.0, 28.0), "base_demand": 16},
    }
    
    cat_keys = list(categories.keys())
    products = []
    
    for i in range(1, num_products + 1):
        cat = np.random.choice(cat_keys)
        cat_info = categories[cat]
        
        shelf_life = int(np.random.randint(cat_info["shelf_life"][0], cat_info["shelf_life"][1]))
        price = round(float(np.random.uniform(cat_info["price_range"][0], cat_info["price_range"][1])), 2)
        velocity_factor = round(float(np.random.uniform(0.5, 2.0)), 2) # fast vs slow mover
        
        products.append({
            "product_id": f"MED-{i:03d}",
            "product_name": f"{cat[:4].upper()}-Compound-{i:03d}",
            "category": cat,
            "unit_price": price,
            "typical_shelf_life_days": shelf_life,
            "base_daily_demand": int(cat_info["base_demand"] * velocity_factor)
        })
        
    return pd.DataFrame(products)

def generate_batches_and_sales(products_df, start_date="2022-01-01", days=730):
    tables = _simulate(products_df, start_date, days)
    return tables["batches"], tables["sales"]


def generate_dataset(seed=42, num_products=80, days=730, start_date="2022-01-01"):
    set_seed(seed)
    products = generate_products(num_products)
    return {"products": products, **_simulate(products, start_date, days)}


def _simulate(products_df, start_date, days):
    if days <= 0 or products_df.empty:
        raise ValueError("days and product count must be positive")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start_dt + timedelta(days=d) for d in range(days)]
    
    batches = []
    sales = []
    daily_sales = []
    snapshots = []
    
    batch_counter = 1
    sale_counter = 1
    
    # Active batches per product: list of dicts {batch_id, exp_date, qty_rem}
    active_inventory = {row["product_id"]: [] for _, row in products_df.iterrows()}
    
    for current_date in date_range:
        day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
        month = current_date.month
        
        # Seasonality factors:
        # Respiratory & Antipyretics spike in Winter (Dec-Feb) and Monsoon (July-Aug)
        winter_factor = 1.4 if month in [12, 1, 2] else 1.0
        monsoon_factor = 1.3 if month in [7, 8] else 1.0
        weekend_factor = 1.2 if day_of_week in [5, 6] else 1.0
        
        for prod in products_df.to_dict("records"):
            pid = prod["product_id"]
            cat = prod["category"]
            base = prod["base_daily_demand"]
            
            # Category-specific seasonality
            season_mult = 1.0
            if cat in ["Respiratory", "Antipyretics"]:
                season_mult = max(winter_factor, monsoon_factor)
            elif cat == "Vitamins_Supplements":
                season_mult = 1.2 if month in [1, 2] else 1.0 # New year fitness spikes
            
            # Reorder below ten days of baseline demand; unsolicited deliveries can overstock.
            total_remaining = sum(b["qty_remaining"] for b in active_inventory[pid] if b["expiry_dt"] > current_date)
            reorder_threshold = base * 10
            
            if total_remaining < reorder_threshold or np.random.rand() < 0.03: # periodic batch restock
                shelf_life = prod["typical_shelf_life_days"]
                # Stress scenarios, not estimates of real pharmacy prevalence.
                short_dated = np.random.rand() < 0.3
                actual_shelf_life = (int(np.random.randint(7, 61)) if short_dated
                                    else int(shelf_life * np.random.uniform(0.6, 1.0)))
                exp_date = current_date + timedelta(days=actual_shelf_life)
                batch_size = int(base * np.random.randint(25, 60))
                if np.random.rand() < 0.15:
                    batch_size *= 3
                
                batch_obj = {
                    "batch_id": f"BAT-{batch_counter:05d}",
                    "product_id": pid,
                    "manufacture_date": (exp_date - timedelta(days=int(shelf_life))).strftime("%Y-%m-%d"),
                    "received_date": current_date.strftime("%Y-%m-%d"),
                    "expiry_date": exp_date.strftime("%Y-%m-%d"),
                    "expiry_dt": exp_date,
                    "quantity_received": batch_size,
                    "qty_remaining": batch_size,
                    "sold_out_date": None,
                }
                batches.append(batch_obj)
                active_inventory[pid].append(batch_obj)
                batch_counter += 1
            
            # Simulate daily demand with Poisson distribution
            expected_demand = max(1.0, base * season_mult * weekend_factor)
            daily_demand = np.random.poisson(lam=expected_demand)
            
            # Sell from the earliest-expiring valid batch (FEFO policy).
            active_batches = [b for b in active_inventory[pid] if b["expiry_dt"] > current_date and b["qty_remaining"] > 0]
            active_batches.sort(key=lambda x: x["expiry_dt"]) # oldest expiry first
            
            units_to_dispense = daily_demand
            for b in active_batches:
                if units_to_dispense <= 0:
                    break
                qty_sold = min(b["qty_remaining"], units_to_dispense)
                b["qty_remaining"] -= qty_sold
                units_to_dispense -= qty_sold
                if b["qty_remaining"] == 0:
                    b["sold_out_date"] = current_date.strftime("%Y-%m-%d")
                
                sales.append({
                    "sale_id": f"SAL-{sale_counter:07d}",
                    "sale_date": current_date.strftime("%Y-%m-%d"),
                    "product_id": pid,
                    "batch_id": b["batch_id"],
                    "quantity_sold": qty_sold,
                    "unit_price": prod["unit_price"]
                })
                sale_counter += 1
            daily_sales.append({
                "date": current_date.strftime("%Y-%m-%d"), "product_id": pid,
                "quantity_sold": daily_demand - units_to_dispense,
                "simulated_demand": daily_demand, "unmet_demand": units_to_dispense,
            })
            # Sunday end-of-day snapshots contain only information available then.
            if day_of_week == 6:
                for b in active_batches:
                    if b["qty_remaining"] > 0:
                        snapshots.append({
                            "as_of_date": current_date.strftime("%Y-%m-%d"),
                            "batch_id": b["batch_id"], "product_id": pid,
                            "quantity_remaining": b["qty_remaining"],
                        })

    # Format batches dataframe
    batches_df = pd.DataFrame([
        {
            "batch_id": b["batch_id"],
            "product_id": b["product_id"],
            "manufacture_date": b["manufacture_date"],
            "received_date": b["received_date"],
            "expiry_date": b["expiry_date"],
            "quantity_received": b["quantity_received"],
            "quantity_remaining": b["qty_remaining"],
            "outcome_status": ("sold_out" if b["sold_out_date"] else
                               "expired" if b["expiry_dt"] <= date_range[-1] else "censored"),
            "outcome_date": (b["sold_out_date"] or
                             (b["expiry_date"] if b["expiry_dt"] <= date_range[-1] else None)),
            "expiry_risk": (0 if b["sold_out_date"] else
                            int(b["qty_remaining"] > 0.2 * b["quantity_received"])
                            if b["expiry_dt"] <= date_range[-1] else None),
        } for b in batches
    ])
    
    batches_df["expiry_risk"] = batches_df.expiry_risk.astype("Int64")
    return {
        "batches": batches_df,
        "sales": pd.DataFrame(sales, columns=["sale_id", "sale_date", "product_id",
                                              "batch_id", "quantity_sold", "unit_price"]),
        "daily_sales": pd.DataFrame(daily_sales),
        "inventory_snapshots": pd.DataFrame(snapshots, columns=["as_of_date", "batch_id",
                                                              "product_id", "quantity_remaining"]),
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pharmacy dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--products", type=int, default=80, help="Number of products")
    parser.add_argument("--days", type=int, default=730, help="Number of sales days")
    parser.add_argument("--start-date", default="2022-01-01", help="First date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    try:
        tables = generate_dataset(args.seed, args.products, args.days, args.start_date)
    except ValueError as exc:
        parser.error(str(exc))
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {"schema_version": 2, "seed": args.seed, "products": args.products,
                "days": args.days, "start_date": args.start_date, "risk_threshold": 0.2,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "files": {}}
    for name, table in tables.items():
        path = os.path.join(args.output_dir, name + ".csv")
        table.to_csv(path, index=False, lineterminator="\n")
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        manifest["files"][name + ".csv"] = {"rows": len(table), "sha256": digest}
        print(f"{name}: {len(table):,} rows -> {path}")
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

if __name__ == "__main__":
    main()
