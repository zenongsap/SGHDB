"""
Delta/Incremental Update Script for HDB Resale Data from data.gov.sg
--------------------------------------------------------------------

This script performs the SAME preprocessing and transformation steps as:
- 1-VisualStudio_DataPreproceses.ipynb (preprocessing)
- 3-Snowflake_Data-Transformation.ipynb (transformation & feature engineering)

Steps performed (matching your notebooks):
1. Fetch delta data from data.gov.sg API (only new records since last sync)
2. Data Preprocessing (from 1-VisualStudio_DataPreproceses.ipynb):
   - Force headers uppercase
   - Calculate AGE from REMAINING_LEASE or LEASE_COMMENCE_DATE
   - Convert text columns to uppercase
   - Split MONTH into YEAR and MONTH_NUM
   - Filter columns (exclude unwanted columns)
   - Remove duplicates
   - Convert numeric columns to int
3. Data Transformation (from 3-Snowflake_Data-Transformation.ipynb):
   - Remove rows with FLAT_TYPE in ['MULTI GENERATION', '1 ROOM', '2 ROOM']
   - Remove FLAT_MODEL column
   - Outlier handling (cap RESALE_PRICE at 0.5th and 99.5th percentiles)
   - Create IS_OUTLIERS flag
   - Feature engineering: AGE_GROUP, PRICE_TIER, SEASON, STOREY_NUMERIC, PRICE_PER_SQM
4. Merge with existing data and save to raw_data_main.csv

Usage:
    # First run: full sync
    python scripts/update_hdb_data_delta.py --full
    
    # Subsequent runs: incremental sync (only new data since last sync)
    python scripts/update_hdb_data_delta.py
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_BASE = "https://data.gov.sg/api/action/datastore_search"
CSV_FILE = "raw_data_main.csv"
LAST_SYNC_FILE = ".last_sync_date.txt"
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def get_last_sync_date() -> str | None:
    """
    Returns the last synced month (YYYY-MM format) or None if never synced.
    If .last_sync_date.txt doesn't exist, tries to infer from raw_data_main.csv.
    """
    sync_file = PROJECT_ROOT / LAST_SYNC_FILE
    if sync_file.exists():
        with open(sync_file, "r") as f:
            date_str = f.read().strip()
            if date_str:
                return date_str
    
    # If sync file doesn't exist, check CSV file for latest month
    csv_file = PROJECT_ROOT / CSV_FILE
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # Read sample to check structure
            if 'YEAR' in df.columns and 'MONTH_NUM' in df.columns:
                # Read full file to get latest date
                df_full = pd.read_csv(csv_file)
                if len(df_full) > 0:
                    # Get the latest year and month
                    latest_year = df_full['YEAR'].max()
                    latest_month_num = df_full[df_full['YEAR'] == latest_year]['MONTH_NUM'].max()
                    latest_month = f"{int(latest_year)}-{int(latest_month_num):02d}"
                    print(f"📅 Inferred last sync date from CSV: {latest_month}")
                    return latest_month
        except Exception as e:
            print(f"⚠️ Could not infer sync date from CSV: {e}")
    
    return None


def save_last_sync_date(month: str):
    """Saves the latest month we've synced."""
    sync_file = PROJECT_ROOT / LAST_SYNC_FILE
    with open(sync_file, "w") as f:
        f.write(month)


def fetch_all_records_since(month_start: str | None = None):
    """
    Fetches records from data.gov.sg API efficiently.
    
    Strategy:
    - If month_start is provided (incremental sync): Fetch in REVERSE chronological order
      (newest first) and stop once we hit records older than month_start. This is much faster!
    - If no month_start (full sync): Fetch all records in chronological order
    
    Returns: DataFrame with fetched records (raw API format), filtered by month_start if provided
    """
    all_records = []
    offset = 0
    limit = 1000  # Max records per request
    max_records_to_fetch = 50000  # Safety limit for incremental sync (fetch max 50K records)
    
    if month_start:
        # OPTIMIZED: Fetch in reverse order (newest first) and stop when we hit old records
        print(f"📥 Fetching records in reverse chronological order (newest first)...")
        print(f"   Will stop when we reach records older than {month_start}")
        sort_order = "month desc"  # Reverse chronological order
        stop_when_older_than = month_start
    else:
        # Full sync: fetch all records in chronological order
        print(f"📥 Fetching all records (full sync)...")
        sort_order = "month"  # Chronological order
        stop_when_older_than = None
        max_records_to_fetch = None  # No limit for full sync
    
    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        # Check if we've hit the limit for incremental sync
        if max_records_to_fetch and len(all_records) >= max_records_to_fetch:
            print(f"  ℹ️ Reached safety limit of {max_records_to_fetch:,} records for incremental sync")
            break
        
        params = {
            "resource_id": DATASET_ID,
            "limit": limit,
            "offset": offset,
            "sort": sort_order
        }
        
        try:
            response = requests.get(API_BASE, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                error_msg = data.get('error', 'Unknown error')
                print(f"❌ API error: {error_msg}")
                break
            
            records = data["result"]["records"]
            if not records:
                print(f"  ℹ️ No more records at offset {offset}")
                break
            
            # Show sample of what we're getting
            if offset == 0 and records:
                sample_month = records[0].get("month", "")
                print(f"  📅 Sample month from first batch: {sample_month}")
            
            # For incremental sync: Check if we've reached records older than month_start
            if stop_when_older_than and records:
                # Check the last record in this batch (oldest in reverse order)
                oldest_in_batch = records[-1].get("month", "")
                if oldest_in_batch and oldest_in_batch < stop_when_older_than:
                    # We've reached records older than what we need
                    # Keep only records >= month_start
                    new_records = [r for r in records if r.get("month", "") >= stop_when_older_than]
                    if new_records:
                        all_records.extend(new_records)
                        print(f"  ✅ Fetched {len(new_records)} records (total: {len(all_records):,})...")
                    print(f"  🛑 Reached records older than {stop_when_older_than}. Stopping fetch.")
                    break
                else:
                    # All records in this batch are new enough
                    all_records.extend(records)
                    print(f"  ✅ Fetched {len(records)} records (total: {len(all_records):,})...")
            else:
                # Full sync: add all records
                all_records.extend(records)
                print(f"  ✅ Fetched {len(records)} records (total: {len(all_records):,})...")
            
            # Check if we've reached the end (API returns fewer than limit)
            if len(records) < limit:
                print(f"  ℹ️ Reached end of data (got {len(records)} < {limit} records)")
                break
            
            offset += limit
            iteration += 1
            
            # Progress indicator every 10 iterations
            if iteration % 10 == 0:
                print(f"  📊 Progress: {len(all_records):,} records fetched so far...")
            
        except requests.RequestException as e:
            print(f"❌ Error fetching data at offset {offset}: {e}")
            break
    
    if not all_records:
        print("⚠️ No records found from API.")
        return None
    
    df = pd.DataFrame(all_records)
    print(f"✅ Total records fetched from API: {len(df):,}")
    
    # Final filter by month_start in Python (double-check, in case reverse order didn't work)
    if month_start and "month" in df.columns:
        initial_count = len(df)
        # Ensure month column is string for comparison
        df["month"] = df["month"].astype(str)
        df_filtered = df[df["month"] >= month_start].copy()
        filtered_count = len(df_filtered)
        
        if filtered_count < initial_count:
            print(f"📊 Final filter: {filtered_count:,} records with month >= {month_start} (removed {initial_count - filtered_count:,} older records)")
        
        if len(df_filtered) == 0:
            print(f"⚠️ No records found with month >= {month_start}")
            if len(df) > 0:
                available_months = sorted(df['month'].unique())
                print(f"   Available months in fetched data: {available_months[-10:]}")
            return None
        
        return df_filtered
    
    return df


# ====================
# PREPROCESSING (from 1-VisualStudio_DataPreproceses.ipynb)
# ====================

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses raw API data following 1-VisualStudio_DataPreproceses.ipynb
    
    Steps:
    1. Force headers uppercase
    2. Calculate AGE from REMAINING_LEASE or LEASE_COMMENCE_DATE
    3. Convert text columns to uppercase
    4. Split MONTH into YEAR and MONTH_NUM
    5. Filter columns (exclude unwanted columns)
    6. Remove duplicates
    7. Convert numeric columns to int
    """
    df = df.copy()
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing (from 1-VisualStudio_DataPreproceses.ipynb)")
    print("=" * 60)
    
    # Step 1.1: Force headers uppercase
    print("1.1. Converting column headers to uppercase...")
    df.columns = [col.strip().upper() for col in df.columns]
    
    # Step 1.2: Calculate AGE from REMAINING_LEASE or LEASE_COMMENCE_DATE
    print("1.2. Calculating AGE...")
    current_year = datetime.now().year
    
    if 'REMAINING_LEASE' in df.columns:
        # Extract numeric years from REMAINING_LEASE (e.g., "94 years 10 months" -> 94.83)
        def extract_lease_years(lease_str):
            try:
                if pd.isna(lease_str):
                    return None
                parts = str(lease_str).split(' years')
                if len(parts) > 0:
                    years = float(parts[0])
                    # Try to extract months if present
                    if 'months' in str(lease_str):
                        months_part = str(lease_str).split('years')[1].split('months')[0].strip()
                        try:
                            months = float(months_part)
                            return years + (months / 12)
                        except:
                            pass
                    return years
            except:
                return None
        
        df['REMAINING_LEASE_NUMERIC'] = df['REMAINING_LEASE'].apply(extract_lease_years)
        df['AGE'] = (99 - df['REMAINING_LEASE_NUMERIC']).fillna(0).astype(int)
        print(f"   ✅ Created AGE from REMAINING_LEASE")
    elif 'LEASE_COMMENCE_DATE' in df.columns:
        # Clean and convert LEASE_COMMENCE_DATE to numeric
        df['LEASE_COMMENCE_DATE'] = pd.to_numeric(df['LEASE_COMMENCE_DATE'], errors='coerce')
        df['AGE'] = (current_year - df['LEASE_COMMENCE_DATE']).fillna(0).astype(int)
        print(f"   ✅ Created AGE from LEASE_COMMENCE_DATE")
    else:
        print("   ⚠️ No lease columns found. AGE cannot be calculated.")
        df['AGE'] = 0
    
    # Step 1.3: Convert text columns to uppercase
    print("1.3. Converting text columns to uppercase...")
    text_columns = ['TOWN', 'FLAT_TYPE', 'STREET_NAME', 'FLAT_MODEL']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
            df[col] = df[col].replace('NAN', pd.NA)
            print(f"   ✅ Converted {col} to uppercase")
    
    # Step 1.4: Split MONTH field into YEAR and MONTH_NUM
    print("1.4. Splitting MONTH field into YEAR and MONTH_NUM...")
    if 'MONTH' in df.columns:
        try:
            df['MONTH'] = pd.to_datetime(df['MONTH'], format='%Y-%m', errors='coerce')
            df['YEAR'] = df['MONTH'].dt.year
            df['MONTH_NUM'] = df['MONTH'].dt.month
            print(f"   ✅ Successfully split MONTH into YEAR and MONTH_NUM")
        except Exception as e:
            print(f"   ❌ Error splitting MONTH field: {e}")
    else:
        print("   ⚠️ MONTH column not found")
    
    # Step 1.5: Convert numeric columns
    print("1.5. Converting numeric columns...")
    if 'RESALE_PRICE' in df.columns:
        df['RESALE_PRICE'] = pd.to_numeric(df['RESALE_PRICE'], errors='coerce').fillna(0).astype(int)
    if 'FLOOR_AREA_SQM' in df.columns:
        df['FLOOR_AREA_SQM'] = pd.to_numeric(df['FLOOR_AREA_SQM'], errors='coerce').fillna(0).astype(int)
    
    # Step 1.6: Remove duplicates
    print("1.6. Removing duplicates...")
    initial_count = len(df)
    # Identify duplicates based on key columns
    key_columns = ['MONTH', 'BLOCK', 'STREET_NAME', 'TOWN', 'FLAT_TYPE']
    key_columns = [col for col in key_columns if col in df.columns]
    if key_columns:
        df = df.drop_duplicates(subset=key_columns, keep='last')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            print(f"   ✅ Removed {duplicates_removed:,} duplicate rows")
        else:
            print("   ✅ No duplicates found")
    
    print(f"✅ Preprocessing complete. Remaining rows: {len(df):,}")
    return df


# ====================
# TRANSFORMATION (from 3-Snowflake_Data-Transformation.ipynb)
# ====================

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms preprocessed data following 3-Snowflake_Data-Transformation.ipynb
    
    Steps:
    1. Data Cleaning:
       - Remove rows with FLAT_TYPE in ['MULTI GENERATION', '1 ROOM', '2 ROOM']
       - Remove FLAT_MODEL column
    2. Outlier Handling:
       - Cap RESALE_PRICE at 0.5th and 99.5th percentiles
       - Create IS_OUTLIERS flag
    3. Feature Engineering:
       - AGE_GROUP (New, Moderate, Old, Very Old)
       - PRICE_TIER (Budget, Mid-range, Premium, Luxury)
       - SEASON (Q1, Q2, Q3, Q4)
       - STOREY_NUMERIC (midpoint from STOREY_RANGE)
       - PRICE_PER_SQM (RESALE_PRICE / FLOOR_AREA_SQM)
    """
    df = df.copy()
    print("\n" + "=" * 60)
    print("STEP 2: Data Transformation (from 3-Snowflake_Data-Transformation.ipynb)")
    print("=" * 60)
    
    # Step 2.1: Data Cleaning
    print("2.1. Data Cleaning...")
    initial_row_count = len(df)
    
    # Remove rows with specified FLAT_TYPEs
    flat_types_to_remove = ['MULTI GENERATION', '1 ROOM', '2 ROOM']
    df = df[~df['FLAT_TYPE'].isin(flat_types_to_remove)]
    rows_removed_ft = initial_row_count - len(df)
    print(f"   ✅ Removed {rows_removed_ft:,} rows with FLAT_TYPE in {flat_types_to_remove}")
    
    # Remove FLAT_MODEL column
    if 'FLAT_MODEL' in df.columns:
        df = df.drop(columns=['FLAT_MODEL'])
        print("   ✅ Dropped 'FLAT_MODEL' column")
    
    # Step 2.2: Outlier Handling (Capping)
    print("2.2. Outlier Handling (Capping RESALE_PRICE)...")
    if 'RESALE_PRICE' in df.columns and len(df) > 0:
        lower_q = df['RESALE_PRICE'].quantile(0.005)
        upper_q = df['RESALE_PRICE'].quantile(0.995)
        
        # Flag outliers BEFORE capping
        df['IS_OUTLIERS'] = ((df['RESALE_PRICE'] < lower_q) | (df['RESALE_PRICE'] > upper_q)).astype(int)
        
        # Cap resale prices
        df['RESALE_PRICE'] = df['RESALE_PRICE'].clip(lower=lower_q, upper=upper_q)
        print(f"   ✅ Capped RESALE_PRICE at {lower_q:,.0f} (0.5th percentile) and {upper_q:,.0f} (99.5th percentile)")
        print(f"   ✅ Created IS_OUTLIERS flag: {df['IS_OUTLIERS'].sum():,} outliers flagged")
    
    # Step 2.3: Feature Engineering
    print("2.3. Feature Engineering...")
    
    # AGE_GROUP
    if 'AGE' in df.columns:
        bins = [0, 5, 15, 30, float('inf')]
        labels = ['New', 'Moderate', 'Old', 'Very Old']
        df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
        print(f"   ✅ Created AGE_GROUP")
    
    # PRICE_TIER
    if 'RESALE_PRICE' in df.columns and len(df) > 0:
        bins = df['RESALE_PRICE'].quantile([0, 0.25, 0.75, 0.95, 1.0])
        labels = ['Budget', 'Mid-range', 'Premium', 'Luxury']
        df['PRICE_TIER'] = pd.cut(df['RESALE_PRICE'], bins=bins, labels=labels, include_lowest=True)
        print(f"   ✅ Created PRICE_TIER")
    
    # SEASON
    if 'MONTH_NUM' in df.columns:
        def get_season(month):
            if pd.isna(month):
                return None
            if 1 <= month <= 3:
                return 'Q1'
            elif 4 <= month <= 6:
                return 'Q2'
            elif 7 <= month <= 9:
                return 'Q3'
            elif 10 <= month <= 12:
                return 'Q4'
            return None
        
        df['SEASON'] = df['MONTH_NUM'].apply(get_season)
        print(f"   ✅ Created SEASON from MONTH_NUM")
    
    # STOREY_NUMERIC
    if 'STOREY_RANGE' in df.columns:
        def get_middle_storey(storey_range):
            """Extracts the numeric midpoint from a storey range string."""
            try:
                if pd.isna(storey_range):
                    return None
                parts = str(storey_range).split(' TO ')
                if len(parts) == 2:
                    lower = int(parts[0])
                    upper = int(parts[1])
                    return (lower + upper) / 2
            except (ValueError, IndexError):
                pass
            return None
        
        df['STOREY_NUMERIC'] = df['STOREY_RANGE'].apply(get_middle_storey)
        print(f"   ✅ Created STOREY_NUMERIC from STOREY_RANGE")
    
    # PRICE_PER_SQM
    if 'RESALE_PRICE' in df.columns and 'FLOOR_AREA_SQM' in df.columns:
        df['PRICE_PER_SQM'] = df['RESALE_PRICE'] / df['FLOOR_AREA_SQM'].replace(0, np.nan)
        print(f"   ✅ Created PRICE_PER_SQM")
    
    # Step 2.4: Filter columns (match your raw_data_main.csv format)
    print("2.4. Filtering columns to match raw_data_main.csv format...")
    exclude_cols = ['STREET_NAME', 'SOURCE_FILE', 'LEASE_COMMENCE_DATE', 'MONTH', 
                    'REMAINING_LEASE', 'REMAINING_LEASE_YEARS', 'BLOCK', 'REMAINING_LEASE_NUMERIC']
    
    # Keep only columns that match your raw_data_main.csv format
    expected_cols = ['TOWN', 'FLAT_TYPE', 'STOREY_RANGE', 'FLOOR_AREA_SQM', 'RESALE_PRICE', 
                     'AGE', 'YEAR', 'MONTH_NUM', 'IS_OUTLIERS', 'AGE_GROUP', 'PRICE_TIER', 
                     'SEASON', 'STOREY_NUMERIC', 'PRICE_PER_SQM']
    
    keep_cols = [col for col in df.columns if col in expected_cols and col not in exclude_cols]
    df = df[keep_cols].copy()
    print(f"   ✅ Filtered to {len(keep_cols)} columns matching raw_data_main.csv format")
    
    print(f"✅ Transformation complete. Final rows: {len(df):,}")
    return df


def merge_with_existing(new_df: pd.DataFrame, existing_file: Path) -> pd.DataFrame:
    """
    Merges new records with existing CSV file, removing duplicates.
    """
    csv_path = PROJECT_ROOT / existing_file
    
    if not csv_path.exists():
        print(f"\n📁 Existing file not found. Creating new file with {len(new_df):,} records.")
        return new_df
    
    print(f"\n📁 Loading existing data from {existing_file}...")
    df_existing = pd.read_csv(csv_path)
    print(f"   Existing records: {len(df_existing):,}")
    
    # Concatenate and remove duplicates
    df_combined = pd.concat([df_existing, new_df], ignore_index=True)
    
    # Remove duplicates based on key columns
    key_columns = ['YEAR', 'MONTH_NUM', 'TOWN', 'FLAT_TYPE', 'STOREY_RANGE', 'FLOOR_AREA_SQM']
    key_columns = [col for col in key_columns if col in df_combined.columns]
    
    if key_columns:
        initial_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=key_columns, keep='last')
        duplicates_removed = initial_count - len(df_combined)
        if duplicates_removed > 0:
            print(f"   ✅ Removed {duplicates_removed:,} duplicate rows after merge")
    
    # Sort by YEAR and MONTH_NUM
    if 'YEAR' in df_combined.columns and 'MONTH_NUM' in df_combined.columns:
        df_combined = df_combined.sort_values(by=['YEAR', 'MONTH_NUM']).reset_index(drop=True)
    
    new_rows_added = len(df_combined) - len(df_existing)
    print(f"   ✅ After merge: {len(df_combined):,} total records ({new_rows_added:,} new records added)")
    
    return df_combined


def update_data(incremental: bool = True):
    """
    Main function to update HDB data with full preprocessing and transformation.
    
    Args:
        incremental: If True, only fetch records since last sync.
                    If False, fetch all records (full sync).
    """
    print("=" * 70)
    print("🚀 HDB Data Update Script")
    print("=" * 70)
    print(f"Mode: {'Incremental (Delta) Sync' if incremental else 'Full Sync'}")
    print("=" * 70)
    
    last_sync_month = get_last_sync_date() if incremental else None
    
    # Calculate the month to fetch from
    if incremental and last_sync_month:
        print(f"📅 Last sync date: {last_sync_month}")
        # For incremental sync, fetch data starting from the NEXT month
        # If last sync was 2025-08, we want data from 2025-09 onwards
        try:
            year, month = last_sync_month.split('-')
            year, month = int(year), int(month)
            # Calculate next month
            if month == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month + 1
            fetch_from_month = f"{next_year}-{next_month:02d}"
            print(f"📥 Fetching data from {fetch_from_month} onwards...")
        except Exception as e:
            # If parsing fails, use the original date
            print(f"⚠️ Could not parse last sync date, using as-is: {e}")
            fetch_from_month = last_sync_month
    else:
        fetch_from_month = None
        if not incremental:
            print("📥 Performing full sync (fetching all data)...")
    
    # Step 1: Fetch raw data from API
    df_new = fetch_all_records_since(fetch_from_month)
    
    if df_new is None or len(df_new) == 0:
        if last_sync_month:
            print(f"\n✅ No new data to update since {last_sync_month}.")
        else:
            print("\n✅ No new data to update.")
        return None
    
    # Step 2: Preprocess (from 1-VisualStudio_DataPreproceses.ipynb)
    df_preprocessed = preprocess_raw_data(df_new)
    
    # Step 3: Transform (from 3-Snowflake_Data-Transformation.ipynb)
    df_transformed = transform_data(df_preprocessed)
    
    if df_transformed is None or len(df_transformed) == 0:
        print("\n⚠️ No data remaining after transformation.")
        return None
    
    # Step 4: Merge with existing data
    df_final = merge_with_existing(df_transformed, CSV_FILE)
    
    # Step 5: Save updated CSV
    csv_path = PROJECT_ROOT / CSV_FILE
    df_final.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(df_final):,} records to {CSV_FILE}")
    
    # Step 6: Update last sync date (use the latest month from new data)
    if 'MONTH' in df_new.columns:
        # Convert MONTH back to string if it's datetime
        if pd.api.types.is_datetime64_any_dtype(df_new['MONTH']):
            latest_month = df_new['MONTH'].max().strftime('%Y-%m')
        else:
            latest_month = str(df_new['MONTH'].max())
        save_last_sync_date(latest_month)
        print(f"✅ Last sync date updated to: {latest_month}")
    elif 'YEAR' in df_transformed.columns and 'MONTH_NUM' in df_transformed.columns:
        # Fallback: construct from YEAR and MONTH_NUM
        latest_row = df_transformed.sort_values(by=["YEAR", "MONTH_NUM"], ascending=[True, True]).iloc[-1]
        latest_month = f"{int(latest_row['YEAR'])}-{int(latest_row['MONTH_NUM']):02d}"

        save_last_sync_date(latest_month)
        print(f"✅ Last sync date updated to: {latest_month}")
    
    print("\n" + "=" * 70)
    print("🎉 Data update complete!")
    print("=" * 70)
    
    return df_final


if __name__ == "__main__":
    # Command line argument: --full for full sync, default is incremental
    full_sync = "--full" in sys.argv
    
    try:
        result = update_data(incremental=not full_sync)
        # Exit successfully even if no new data (this is normal)
        if result is None:
            print("\n✅ Script completed successfully (no new data to update).")
            sys.exit(0)
        else:
            print("\n✅ Script completed successfully (data updated).")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️ Update cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during update: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)