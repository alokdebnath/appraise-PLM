#!/usr/bin/env python3
"""
Fix CSV Files by Properly Quoting Fields

This script fixes the CSV parsing issues by:
1. Properly quoting all fields
2. Handling malformed rows
3. Creating clean, properly formatted CSV files
"""

import csv
import re
from pathlib import Path

def clean_and_fix_csv(input_file, output_file):
    """Clean and fix a CSV file by properly handling commas in text fields."""
    print(f"\n=== Fixing {input_file.name} -> {output_file.name} ===")
    
    fixed_rows = 0
    skipped_rows = 0
    total_rows = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            # Read the original file line by line to handle malformed rows
            lines = infile.readlines()
            
            # Write header
            header = lines[0].strip().split(',')
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            expected_columns = len(header)
            
            print(f"Expected columns: {expected_columns}")
            print(f"Header: {header}")
            
            # Process data rows
            for line_num, line in enumerate(lines[1:], 2):
                total_rows += 1
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                
                # Try to parse the line as CSV
                try:
                    # Use CSV reader to properly handle quoted fields
                    row = list(csv.reader([line]))[0]
                    
                    # Check if we have the right number of columns
                    if len(row) == expected_columns:
                        writer.writerow(row)
                        fixed_rows += 1
                    else:
                        # Try to fix malformed rows by joining extra columns
                        if len(row) > expected_columns:
                            # Join the extra columns into the last expected column
                            fixed_row = row[:expected_columns-1]
                            remaining_content = ','.join(row[expected_columns-1:])
                            fixed_row.append(remaining_content)
                            writer.writerow(fixed_row)
                            fixed_rows += 1
                        else:
                            # Skip rows with too few columns
                            skipped_rows += 1
                            if skipped_rows <= 5:
                                print(f"Skipping line {line_num}: too few columns ({len(row)} < {expected_columns})")
                
                except Exception as e:
                    skipped_rows += 1
                    if skipped_rows <= 5:
                        print(f"Skipping line {line_num}: parsing error - {e}")
                
                # Show progress
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False
    
    print(f"\nProcessing complete:")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Successfully fixed: {fixed_rows}")
    print(f"  Skipped: {skipped_rows}")
    
    return True

def verify_fixed_file(filepath):
    """Verify that the fixed file has correct column counts."""
    print(f"\n=== Verifying {filepath.name} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_columns = len(header)
            
            wrong_count = 0
            total_rows = 0
            
            for row in reader:
                total_rows += 1
                if len(row) != expected_columns:
                    wrong_count += 1
                    if wrong_count <= 3:
                        print(f"Row {total_rows + 1}: {len(row)} columns instead of {expected_columns}")
            
            print(f"Total rows: {total_rows}")
            print(f"Rows with wrong column count: {wrong_count}")
            
            if wrong_count == 0:
                print("✅ File is properly formatted!")
                return True
            else:
                print("❌ File still has issues")
                return False
                
    except Exception as e:
        print(f"Error verifying file: {e}")
        return False

def main():
    """Fix all CSV files in the current directory."""
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("Found CSV files:", [f.name for f in csv_files])
    
    # Fix each file
    for csv_file in csv_files:
        if csv_file.name.endswith('_fixed.csv'):
            continue  # Skip already fixed files
            
        fixed_name = csv_file.stem + '_fixed.csv'
        if fix_csv_file(csv_file, Path(fixed_name)):
            # Verify the fixed file
            verify_fixed_file(Path(fixed_name))
    
    print("\n" + "="*50)
    print("FIXING COMPLETE")
    print("="*50)
    print("\nFixed files created:")
    for csv_file in csv_files:
        if not csv_file.name.endswith('_fixed.csv'):
            fixed_name = csv_file.stem + '_fixed.csv'
            if Path(fixed_name).exists():
                print(f"  {csv_file.name} -> {fixed_name}")
    
    print("\nRECOMMENDATIONS:")
    print("1. Use the '_fixed.csv' files for your analysis")
    print("2. These files are properly quoted and can be parsed with any CSV reader")
    print("3. Use pandas.read_csv() or Python's csv module to read the fixed files")

def fix_csv_file(input_file, output_file):
    """Wrapper function to handle the fixing process."""
    return clean_and_fix_csv(input_file, output_file)

if __name__ == "__main__":
    main()
