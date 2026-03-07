"""
Shared utilities for data analysis and exploration.
Run this to get a comprehensive overview of the training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent / "Challenge2"

def load_all_data():
    """Load all datasets and return them."""
    print("Loading datasets...")
    
    plis = pd.read_csv(
        BASE_DIR / "plis_training.csv",
        sep='\t', low_memory=False,
        dtype={'eclass': str, 'nace_code': str, 'secondary_nace_code': str}
    )
    plis['orderdate'] = pd.to_datetime(plis['orderdate'])
    
    customers = pd.read_csv(
        BASE_DIR / "customer_test.csv",
        sep='\t',
        dtype={'nace_code': str, 'secondary_nace_code': str}
    )
    
    nace = pd.read_csv(
        BASE_DIR / "nace_codes.csv",
        sep='\t', dtype={'nace_code': str}
    )
    
    features = pd.read_csv(
        BASE_DIR / "features_per_sku.csv",
        sep='\t', low_memory=False
    )
    
    return plis, customers, nace, features


def explore_data(plis, customers, nace, features):
    """Print comprehensive data exploration."""
    
    print("\n" + "="*70)
    print("DATA EXPLORATION REPORT")
    print("="*70)
    
    # Training data overview
    print("\n--- TRAINING DATA (plis_training.csv) ---")
    print(f"Rows: {len(plis):,}")
    print(f"Unique buyers: {plis['legal_entity_id'].nunique():,}")
    print(f"Unique orders (set_id): {plis['set_id'].nunique():,}")
    print(f"Unique SKUs: {plis['sku'].nunique():,}")
    print(f"Unique E-Classes: {plis['eclass'].nunique():,}")
    print(f"Unique Manufacturers: {plis['manufacturer'].nunique():,}")
    print(f"Date range: {plis['orderdate'].min()} to {plis['orderdate'].max()}")
    print(f"\nValue statistics:")
    print(f"  vk_per_item: mean={plis['vk_per_item'].mean():.2f}, median={plis['vk_per_item'].median():.2f}")
    print(f"  quantityvalue: mean={plis['quantityvalue'].mean():.2f}, median={plis['quantityvalue'].median():.2f}")
    
    plis['total_value'] = plis['quantityvalue'].fillna(0) * plis['vk_per_item'].fillna(0)
    print(f"  total_value: mean={plis['total_value'].mean():.2f}, median={plis['total_value'].median():.2f}")
    
    # Year distribution
    print(f"\nOrders by year:")
    print(plis['orderdate'].dt.year.value_counts().sort_index().to_string())
    
    # Buyer activity distribution
    buyer_orders = plis.groupby('legal_entity_id')['set_id'].nunique()
    print(f"\nBuyer order count distribution:")
    print(f"  min={buyer_orders.min()}, q25={buyer_orders.quantile(.25):.0f}, "
          f"median={buyer_orders.median():.0f}, q75={buyer_orders.quantile(.75):.0f}, "
          f"max={buyer_orders.max()}")
    
    # Test customers
    print("\n--- TEST CUSTOMERS (customer_test.csv) ---")
    print(f"Total: {len(customers)}")
    print(f"Tasks: {customers['task'].value_counts().to_dict()}")
    
    warm = customers[customers['task'] == 'predict future']
    cold = customers[customers['task'] == 'cold start']
    
    print(f"\nWarm-start buyers NACE codes:")
    print(warm['nace_code'].value_counts().head(10).to_string())
    print(f"\nCold-start buyers NACE codes:")
    print(cold['nace_code'].value_counts().head(10).to_string())
    
    # NACE codes
    print(f"\n--- NACE CODES ---")
    print(f"Total codes: {len(nace)}")
    print(f"Top-level sections: {nace['toplevel_section'].nunique()}")
    
    # Features
    print(f"\n--- FEATURES PER SKU ---")
    print(f"Total rows: {len(features):,}")
    print(f"Unique SKUs: {features['sku'].nunique():,}")
    print(f"Unique feature keys: {features['key'].nunique():,}")
    print(f"Unique safe_synonyms: {features['safe_synonym'].nunique():,}")
    
    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70)


if __name__ == "__main__":
    plis, customers, nace, features = load_all_data()
    explore_data(plis, customers, nace, features)
