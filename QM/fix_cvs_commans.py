#!/usr/bin/env python3
"""
fix_csv_commas.py - Fix unescaped commas in institute_country_database.csv
"""

import csv
import os

def fix_csv_commas(input_file='institute_country_database.csv', output_file='institute_country_database_fixed.csv'):
    """
    Fix CSV file with unescaped commas in institute names
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to write the fixed CSV file
    """
    print(f"Fixing unescaped commas in {input_file}...")
    
    fixed_rows = []
    problem_lines = []
    
    # Read the original file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:  # Skip empty lines
            fixed_rows.append([''])
            continue
            
        # Count commas in the line
        comma_count = line.count(',')
        
        if comma_count == 0:
            # This is likely a header with no comma, just add it
            fixed_rows.append([line])
        elif comma_count == 1:
            # This is correctly formatted (Institute,Country)
            institute, country = line.split(',')
            fixed_rows.append([institute, country])
        else:
            # This line has too many commas
            problem_lines.append((i+1, line))  # Store 1-based line number
            
            # Try to fix it - assume format is "Institute with, commas,Country"
            parts = line.split(',')
            country = parts[-1]  # Last part is the country
            institute = ','.join(parts[:-1])  # Join everything else as institute
            
            fixed_rows.append([institute, country])
    
    # Write the fixed file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in fixed_rows:
            if len(row) == 1 and not row[0]:  # Empty line
                f.write('\n')
            else:
                writer.writerow(row)
    
    # Report problems
    if problem_lines:
        print(f"\nFixed {len(problem_lines)} problematic lines:")
        for line_num, line in problem_lines:
            print(f"  Line {line_num}: {line}")
    
    print(f"\nFixed file written to {output_file}")
    print("\nTo replace the original file, run:")
    print(f"mv {output_file} {input_file}")

if __name__ == "__main__":
    # Check if the file exists
    if not os.path.exists('institute_country_database.csv'):
        print("Error: institute_country_database.csv not found")
        exit(1)
        
    fix_csv_commas()