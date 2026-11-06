# Data Catalog & Licensing

## Data Policy
As specified in the assessment requirements:
1. No confidential, personal, or live patient healthcare records are stored or processed.
2. All records are either **synthetically generated** via verifiable statistical processes or sourced from open, properly licensed datasets.

## Synthetic Dataset Generation
The data is generated using `src/data_generator.py` with fixed random seed `42`.

### Generated Relational Entities:
1. **`products.csv`**:
   - `product_id`: Unique identifier (e.g., `MED-001`)
   - `product_name`: Generic / brand formulation
   - `category`: Therapeutic class (e.g., Antibiotics, Analgesics, Antipyretics, Cardiovascular, Vitamins)
   - `unit_price`: Base selling price
   - `typical_shelf_life_days`: Standard expiry window from manufacturing

2. **`batches.csv`**:
   - `batch_id`: Batch lot identifier
   - `product_id`: Foreign key referencing `products.csv`
   - `manufacture_date`: Batch production date
   - `expiry_date`: Batch expiration date
   - `quantity_received`: Stock initial volume
   - `quantity_remaining`: Stock currently on pharmacy shelf

3. **`sales.csv`**:
   - `sale_id`: Unique transaction identifier
   - `sale_date`: Daily timestamp
   - `product_id`: Foreign key referencing `products.csv`
   - `batch_id`: Batch lot allocated via FIFO (First In First Out)
   - `quantity_sold`: Units dispensed
   - `unit_price`: Realized price per unit
