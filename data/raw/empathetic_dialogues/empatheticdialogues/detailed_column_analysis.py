#!/usr/bin/env python3
"""
Detailed Column Analysis for CSV Issues

This script analyzes the specific patterns of column content bleeding and
malformed data in the CSV files.
"""

import csv
import re
from pathlib import Path

def analyze_column_patterns(filepath):
    """Analyze specific patterns of column content bleeding and malformed data."""
    print(f"\n=== Detailed Analysis of {filepath.name} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_columns = len(header)
            
            print(f"Expected columns: {expected_columns}")
            print(f"Header: {header}")
            
            patterns_found = {
                'column_bleeding': [],  # Content from one column appearing in another
                'repeated_content': [],  # Same content repeated across columns
                'malformed_selfeval': [],  # Selfeval field with wrong format
                'conv_id_in_wrong_column': [],  # conv_id appearing in utterance column
                'extra_columns_with_patterns': []  # Extra columns with identifiable patterns
            }
            
            total_rows = 0
            problematic_rows = 0
            
            for row_num, row in enumerate(reader, 2):
                total_rows += 1
                actual_columns = len(row)
                
                if actual_columns != expected_columns:
                    problematic_rows += 1
                    
                    # Analyze the specific patterns
                    analysis = analyze_row_patterns(row, header, row_num)
                    
                    for pattern_type, details in analysis.items():
                        if details:
                            patterns_found[pattern_type].extend(details)
                
                # Show progress
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
            
            print(f"\nTotal rows: {total_rows}")
            print(f"Problematic rows: {problematic_rows}")
            
            # Report findings
            print("\n" + "="*60)
            print("PATTERN ANALYSIS RESULTS")
            print("="*60)
            
            for pattern_type, instances in patterns_found.items():
                if instances:
                    print(f"\n{pattern_type.upper().replace('_', ' ')}:")
                    print(f"  Found {len(instances)} instances")
                    for i, instance in enumerate(instances[:3]):  # Show first 3 examples
                        print(f"  Example {i+1}:")
                        print(f"    Line: {instance['line']}")
                        print(f"    Issue: {instance['issue']}")
                        if 'details' in instance:
                            print(f"    Details: {instance['details']}")
                        print()
            
            return patterns_found
            
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def analyze_row_patterns(row, header, line_num):
    """Analyze specific patterns in a problematic row."""
    analysis = {
        'column_bleeding': [],
        'repeated_content': [],
        'malformed_selfeval': [],
        'conv_id_in_wrong_column': [],
        'extra_columns_with_patterns': []
    }
    
    expected_columns = len(header)
    actual_columns = len(row)
    
    if actual_columns <= expected_columns:
        return analysis
    
    # Pattern 1: Check for conv_id pattern in wrong columns
    conv_id_pattern = r'hit:\d+_conv:\d+'
    
    # Check if conv_id appears in utterance column (should be column 5, index 5)
    if actual_columns > 5:
        utterance_field = row[5] if len(row) > 5 else ""
        if re.search(conv_id_pattern, utterance_field):
            analysis['conv_id_in_wrong_column'].append({
                'line': line_num,
                'issue': 'conv_id pattern found in utterance column',
                'details': f"Column 6 (utterance): {utterance_field[:100]}..."
            })
    
    # Pattern 2: Check for malformed selfeval (should be column 6, index 6)
    if actual_columns > 6:
        selfeval_field = row[6] if len(row) > 6 else ""
        # Selfeval should be in format like "5|5|5_2|2|5"
        if not re.match(r'^\d+\|\d+\|\d+_\d+\|\d+\|\d+$', selfeval_field):
            analysis['malformed_selfeval'].append({
                'line': line_num,
                'issue': 'malformed selfeval field',
                'details': f"Expected format: '5|5|5_2|2|5', got: {selfeval_field}"
            })
    
    # Pattern 3: Check for repeated content patterns
    for i in range(expected_columns, actual_columns):
        extra_content = row[i]
        if extra_content.strip():
            # Check if this looks like it should be in a specific column
            if re.search(conv_id_pattern, extra_content):
                analysis['repeated_content'].append({
                    'line': line_num,
                    'issue': f'conv_id content in extra column {i+1}',
                    'details': f"Extra column {i+1}: {extra_content[:100]}..."
                })
            elif '|' in extra_content and '_' in extra_content:
                analysis['repeated_content'].append({
                    'line': line_num,
                    'issue': f'selfeval-like content in extra column {i+1}',
                    'details': f"Extra column {i+1}: {extra_content}"
                })
    
    # Pattern 4: Check for column bleeding (content from one column appearing in another)
    if actual_columns > 6:
        # Check if utterance content is mixed with other data
        utterance_field = row[5] if len(row) > 5 else ""
        if ',' in utterance_field and len(utterance_field) > 200:
            # This might be multiple fields concatenated
            analysis['column_bleeding'].append({
                'line': line_num,
                'issue': 'multiple fields concatenated in utterance column',
                'details': f"Utterance field length: {len(utterance_field)} chars, contains commas"
            })
    
    return analysis

def show_sample_problematic_rows(filepath, num_samples=5):
    """Show detailed examples of problematic rows."""
    print(f"\n=== Sample Problematic Rows from {filepath.name} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_columns = len(header)
            
            sample_count = 0
            
            for row_num, row in enumerate(reader, 2):
                if len(row) != expected_columns and sample_count < num_samples:
                    sample_count += 1
                    print(f"\n--- Problematic Row {sample_count} (Line {row_num}) ---")
                    print(f"Expected columns: {expected_columns}, Actual: {len(row)}")
                    
                    for i, (col_name, value) in enumerate(zip(header, row)):
                        print(f"  {col_name}: {value[:100]}{'...' if len(value) > 100 else ''}")
                    
                    # Show extra columns
                    if len(row) > expected_columns:
                        print(f"\n  Extra columns:")
                        for i in range(expected_columns, len(row)):
                            extra_value = row[i]
                            if extra_value.strip():
                                print(f"    Column {i+1}: {extra_value[:100]}{'...' if len(extra_value) > 100 else ''}")
                    
                    print()
                    
    except Exception as e:
        print(f"Error showing samples: {e}")

def main():
    """Run detailed analysis on all CSV files."""
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("Running detailed column pattern analysis...")
    
    for csv_file in csv_files:
        if csv_file.name.endswith('_fixed.csv'):
            continue
            
        # Run pattern analysis
        patterns = analyze_column_patterns(csv_file)
        
        # Show sample problematic rows
        show_sample_problematic_rows(csv_file)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print("1. Column content bleeding: Data from one column appears in another")
    print("2. Repeated content: Same data appears in multiple columns")
    print("3. Malformed fields: Fields contain wrong type of data")
    print("4. Extra columns: Additional columns with mixed content")
    
    print("\nRecommendations:")
    print("1. The CSV files need to be re-processed to properly separate fields")
    print("2. Consider using a different delimiter or proper quoting")
    print("3. Some rows may need manual inspection and correction")

if __name__ == "__main__":
    main()
