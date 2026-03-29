"""
Script to merge all the annotated CSV files (annotated_*.csv) into a single results_all.csv file for analysis.
"""

import pandas as pd
import glob
 
csv_files = glob.glob("annotated_*.csv")
 
if not csv_files:
    print("No annotated CSV files found.")
else:
    print(f"Found {len(csv_files)} file(s):")
    for f in csv_files:
        print(f"  - {f}")
 
    merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    merged.to_csv("results_all.csv", index=False)
    print(f"\nDone — {len(merged)} rows saved to results_all.csv")
    print(f"Columns: {merged.columns.tolist()}")
