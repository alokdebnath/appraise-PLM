#!/usr/bin/env python3
"""
CSV Analysis and Fix Script for EmpatheticDialogues Dataset

This script analyzes the CSV files to identify parsing issues caused by commas
within text fields and provides solutions to fix them.
"""

import csv
import re
import sys
from pathlib import Path

def analyze_csv_file(filepath):
    """Analyze a CSV file for parsing issues."""
    print(f"\n=== Analyzing {filepath.name} ===")
    
    issues = {
        'wrong_field_count': 0,
        'commas_in_text': 0,
        'unquoted_fields': 0,
        'problematic_rows': []
    }
    
    total_rows = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # First, try to read with standard CSV parser
            reader = csv.reader(f)
            header = next(reader)
            expected_fields = len(header)
            print(f"Expected number of fields: {expected_fields}")
            print(f"Header: {header}")
            
            # Reset file pointer
            f.seek(0)
            
            # Read line by line to check for issues
            for line_num, line in enumerate(f, 1):
                if line_num == 1:  # Skip header
                    continue
                    
                total_rows += 1
                
                # Count commas to check field count
                comma_count = line.count(',')
                if comma_count != expected_fields - 1:
                    issues['wrong_field_count'] += 1
                    issues['problematic_rows'].append({
                        'line': line_num,
                        'issue': f'Wrong field count: {comma_count + 1} instead of {expected_fields}',
                        'content': line.strip()
                    })
                
                # Check for unquoted commas in text (not _comma_)
                if re.search(r'[^_],', line):
                    issues['commas_in_text'] += 1
                    if len(issues['problematic_rows']) < 5:  # Limit examples
                        issues['problematic_rows'].append({
                            'line': line_num,
                            'issue': 'Contains unquoted commas in text',
                            'content': line.strip()
                        })
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Total rows: {total_rows}")
    print(f"Rows with wrong field count: {issues['wrong_field_count']}")
    print(f"Rows with commas in text: {issues['commas_in_text']}")
    
    if issues['problematic_rows']:
        print("\nSample problematic rows:")
        for row in issues['problematic_rows'][:3]:
            print(f"Line {row['line']}: {row['issue']}")
            print(f"Content: {row['content'][:100]}...")
            print()
    
    return issues

def fix_csv_file(input_file, output_file):
    """Fix CSV file by properly quoting fields with commas."""
    print(f"\n=== Fixing {input_file.name} -> {output_file.name} ===")
    
    fixed_rows = 0
    total_rows = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            
            for row in reader:
                total_rows += 1
                writer.writerow(row)
                fixed_rows += 1
                
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
    
    except Exception as e:
        print(f"Error fixing file: {e}")
        return False
    
    print(f"Successfully fixed {fixed_rows} rows")
    return True

def main():
    """Main function to analyze and fix all CSV files."""
    csv_dir = Path('.')
    csv_files = list(csv_dir.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("Found CSV files:", [f.name for f in csv_files])
    
    # Analyze all files
    all_issues = {}
    for csv_file in csv_files:
        issues = analyze_csv_file(csv_file)
        if issues:
            all_issues[csv_file.name] = issues
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_problems = 0
    for filename, issues in all_issues.items():
        problems = issues['wrong_field_count'] + issues['commas_in_text']
        total_problems += problems
        print(f"{filename}: {problems} problematic rows")
    
    if total_problems == 0:
        print("\n✅ No parsing issues found! The CSV files appear to be properly formatted.")
        print("The files can be read using standard CSV parsers.")
    else:
        print(f"\n⚠️  Found {total_problems} potential parsing issues across all files.")
        print("\nRECOMMENDATIONS:")
        print("1. Use a proper CSV parser (like Python's csv module) instead of simple comma splitting")
        print("2. The files may need to be re-saved with proper quoting")
        print("3. Consider using pandas.read_csv() with appropriate parameters")
        
        # Offer to fix the files
        response = input("\nWould you like to create fixed versions of the files? (y/n): ")
        if response.lower() == 'y':
            for csv_file in csv_files:
                fixed_name = csv_file.stem + '_fixed.csv'
                fix_csv_file(csv_file, Path(fixed_name))

if __name__ == "__main__":
    main()
