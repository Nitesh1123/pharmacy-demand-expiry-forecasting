"""
Synthetic Pharmacy Dataset Generator
Generates relational tables: products, batches, and sales with realistic seasonal demand,
shelf-lives, and first-expiry-first-out (FEFO) batch inventory tracking.
"""

import os
import argparse
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
    if days <= 0 or products_df.empty:
        raise ValueError("days and product count must be positive")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start_dt + timedelta(days=d) for d in range(days)]
    
    batches = []
    sales = []
    
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
        
        for _, prod in products_df.iterrows():
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
                    "qty_remaining": batch_size
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
                
                sales.append({
                    "sale_id": f"SAL-{sale_counter:07d}",
                    "sale_date": current_date.strftime("%Y-%m-%d"),
                    "product_id": pid,
                    "batch_id": b["batch_id"],
                    "quantity_sold": qty_sold,
                    "unit_price": prod["unit_price"]
                })
                sale_counter += 1
                
    # Format batches dataframe
    batches_df = pd.DataFrame([
        {
            "batch_id": b["batch_id"],
            "product_id": b["product_id"],
            "manufacture_date": b["manufacture_date"],
            "received_date": b["received_date"],
            "expiry_date": b["expiry_date"],
            "quantity_received": b["quantity_received"],
            "quantity_remaining": b["qty_remaining"]
        } for b in batches
    ])
    
    sales_df = pd.DataFrame(sales)
    return batches_df, sales_df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pharmacy dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--products", type=int, default=80, help="Number of products")
    parser.add_argument("--days", type=int, default=730, help="Number of sales days")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.products} products...")
    products_df = generate_products(args.products)
    
    print(f"Simulating {args.days} days of batch lifecycle and sales...")
    batches_df, sales_df = generate_batches_and_sales(products_df, days=args.days)
    
    # Save datasets
    prod_path = os.path.join(args.output_dir, "products.csv")
    bat_path = os.path.join(args.output_dir, "batches.csv")
    sal_path = os.path.join(args.output_dir, "sales.csv")
    
    products_df.to_csv(prod_path, index=False)
    batches_df.to_csv(bat_path, index=False)
    sales_df.to_csv(sal_path, index=False)
    
    print(f"Successfully generated:")
    print(f"- {len(products_df):,} Products -> {prod_path}")
    print(f"- {len(batches_df):,} Batches -> {bat_path}")
    print(f"- {len(sales_df):,} Sales records -> {sal_path}")

if __name__ == "__main__":
    main()
