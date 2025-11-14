#!/usr/bin/env python3
"""
Check Column Count Issues in CSV Files

This script specifically checks if any rows have more columns than the header
due to unquoted commas in text fields.
"""

import csv
from pathlib import Path

def check_column_count(filepath):
    """Check if any rows have wrong number of columns."""
    print(f"\n=== Checking {filepath.name} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_columns = len(header)
            
            print(f"Header has {expected_columns} columns: {header}")
            
            wrong_count_rows = []
            total_rows = 0
            
            for row_num, row in enumerate(reader, 2):  # Start from 2 since we skipped header
                total_rows += 1
                actual_columns = len(row)
                
                if actual_columns != expected_columns:
                    wrong_count_rows.append({
                        'line': row_num,
                        'expected': expected_columns,
                        'actual': actual_columns,
                        'row': row
                    })
                
                # Show progress for large files
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
            
            print(f"\nTotal rows: {total_rows}")
            print(f"Rows with wrong column count: {len(wrong_count_rows)}")
            
            if wrong_count_rows:
                print(f"\n⚠️  PROBLEM FOUND: {len(wrong_count_rows)} rows have wrong column count!")
                print("\nFirst 5 problematic rows:")
                for i, problem in enumerate(wrong_count_rows[:5]):
                    print(f"\nLine {problem['line']}: {problem['actual']} columns instead of {problem['expected']}")
                    print("Row content:")
                    for j, field in enumerate(problem['row']):
                        print(f"  Column {j+1}: {field[:100]}{'...' if len(field) > 100 else ''}")
                
                return False
            else:
                print("✅ All rows have correct column count!")
                return True
                
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def main():
    """Check all CSV files in the current directory."""
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("Checking CSV files for column count issues...")
    
    all_good = True
    for csv_file in csv_files:
        if not check_column_count(csv_file):
            all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✅ SUMMARY: All CSV files have correct column counts!")
        print("The files can be parsed normally with any CSV reader.")
    else:
        print("❌ SUMMARY: Some CSV files have column count issues!")
        print("\nSOLUTIONS:")
        print("1. Use a proper CSV parser (Python csv module, pandas, etc.)")
        print("2. The fields with commas should be quoted in the CSV")
        print("3. Consider re-saving the files with proper CSV formatting")

if __name__ == "__main__":
    main()
