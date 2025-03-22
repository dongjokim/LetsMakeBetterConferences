#!/usr/bin/env python3
"""
Script to process participant information from manually saved text files.
This is step 1 in the data preparation workflow - run this FIRST.
"""

import os
import re
import json
import csv
from tabulate import tabulate
import glob
import requests
import time

# Constants
OUTPUT_DIR = "data/participants"
DATA_DIR = "data/html"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_indico_ids_from_file(filename='listofQMindigo'):
    """
    Load Indico IDs from a file.
    
    Returns:
    - Dictionary mapping years to Indico IDs
    """
    year_to_id = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        year = parts[0]
                        indico_id = parts[1]
                        year_to_id[year] = indico_id
        
        print(f"Loaded {len(year_to_id)} Indico IDs from {filename}")
        return year_to_id
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return {}

def parse_participant_data_file(data_file, year_to_id):
    """
    Parse participant information from a manually saved text file.
    
    Parameters:
    - data_file: Path to the data file
    - year_to_id: Dictionary mapping years to Indico IDs
    
    Returns:
    - Year, Indico ID, and list of participants with their details
    """
    # Extract year from filename (man_YYYY.data)
    year_match = re.search(r'man_(\d{4})\.data', data_file)
    if not year_match:
        print(f"Warning: Could not extract year from filename {data_file}")
        return None, None, []
    
    year = year_match.group(1)
    indico_id = year_to_id.get(year, "unknown")
    
    participants = []
    
    try:
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Skip empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            print(f"Warning: No data found in {data_file}")
            return year, indico_id, []
        
        # Determine format from first line
        header = lines[0].split('\t')
        has_headers = not header[0].isdigit() and not re.match(r'^\d+$', header[0])
        
        start_idx = 1 if has_headers else 0
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            parts = line.split('\t')
            
            # Skip if not enough data
            if len(parts) < 2:
                continue
            
            name = ""
            affiliation = ""
            country = ""
            
            # Extract data based on the format
            if len(parts) >= 3:
                # Format: First Name, Last Name, Affiliation
                first_name = parts[0].strip()
                last_name = parts[1].strip()
                name = f"{first_name} {last_name}"
                affiliation = parts[2].strip() if len(parts) > 2 else ""
            elif len(parts) == 2:
                # Format: Name, Affiliation
                name = parts[0].strip()
                affiliation = parts[1].strip()
            
            # Try to extract country code from affiliation
            country_match = re.search(r'\(([A-Z]{2,3})\)$', affiliation)
            if country_match:
                country = country_match.group(1)
            
            participants.append({
                'name': name,
                'affiliation': affiliation,
                'country': country,
                'year': year,
                'indico_id': indico_id
            })
        
        print(f"Processed {len(participants)} participants from {data_file}")
        return year, indico_id, participants
    
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return year, indico_id, []

def process_all_data_files():
    """
    Process all data files in the data directory.
    
    Returns:
    - Dictionary mapping event identifiers to lists of participants
    """
    year_to_id = load_indico_ids_from_file()
    all_participants = {}
    
    data_files = glob.glob(os.path.join(DATA_DIR, "man_*.data"))
    
    print(f"Found {len(data_files)} data files to process")
    
    for data_file in data_files:
        year, indico_id, participants = parse_participant_data_file(data_file, year_to_id)
        if year and participants:
            event_key = f"{year}-{indico_id}"
            all_participants[event_key] = participants
    
    return all_participants

def create_combined_file(all_participants):
    """
    Create a combined JSON file with all participants.
    
    Parameters:
    - all_participants: Dictionary mapping event identifiers to lists of participants
    """
    output_file = os.path.join(OUTPUT_DIR, "all_participants.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_participants, f, indent=2, ensure_ascii=False)
    
    print(f"Combined participant data saved to: {output_file}")

def create_csv_files(all_participants):
    """
    Create CSV files for each event and a combined CSV file.
    
    Parameters:
    - all_participants: Dictionary mapping event identifiers to lists of participants
    """
    print("Creating CSV files...")
    
    # Create individual CSV files
    for event_key, participants in all_participants.items():
        output_file = os.path.join(OUTPUT_DIR, f"{event_key}_participants.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'affiliation', 'country', 'year', 'indico_id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for participant in participants:
                writer.writerow(participant)
        
        print(f"  Created CSV file for event {event_key}: {output_file}")
    
    # Create combined CSV file
    output_file = os.path.join(OUTPUT_DIR, "all_participants.csv")
    fieldnames = ['year', 'indico_id', 'name', 'affiliation', 'country']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event_key, participants in all_participants.items():
            for participant in participants:
                writer.writerow({
                    'year': participant.get('year', ''),
                    'indico_id': participant.get('indico_id', ''),
                    'name': participant.get('name', ''),
                    'affiliation': participant.get('affiliation', ''),
                    'country': participant.get('country', '')
                })
    
    print(f"  Saved combined participant data to CSV: {output_file}")
    print("CSV files created successfully!")

def print_participant_summary(all_participants):
    """
    Print a summary of participants.
    
    Parameters:
    - all_participants: Dictionary mapping event identifiers to lists of participants
    """
    print("\nParticipant Summary:")
    
    # Create a table with year, indico_id, and participant count
    table_data = []
    total_count = 0
    
    for event_key in sorted(all_participants.keys()):
        count = len(all_participants[event_key])
        total_count += count
        
        # Extract year and Indico ID from the event key
        if '-' in event_key:
            year, indico_id = event_key.split('-', 1)
        else:
            year = event_key
            indico_id = "unknown"
            
        table_data.append([year, indico_id, count])
    
    table_data.append(["Total", "", total_count])
    
    # Print the table
    print(tabulate(table_data, headers=["Year", "Indico ID", "Participants"], tablefmt="grid"))

def lookup_institute_country(institute_name):
    """
    Look up an institution's country using the ROR API
    https://ror.org/about/
    """
    base_url = "https://api.ror.org/organizations"
    
    try:
        # Query ROR API
        params = {"query": institute_name}
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['items']:
                # Get the first (best) match
                best_match = data['items'][0]
                return best_match['country']['country_name']
        
        # Add delay to respect rate limits
        time.sleep(1)
        
    except Exception as e:
        print(f"Error looking up {institute_name}: {e}")
    
    return 'Unknown'

def extract_country_code(affiliation):
    """Extract country from affiliation string"""
    if not affiliation:
        return 'Unknown'
    
    # First check for country code in parentheses
    match = re.search(r'\((..)\)$', affiliation)
    if match:
        code_map = {
            'US': 'United States',
            'UK': 'United Kingdom',
            'PL': 'Poland',
            # ... (rest of code map)
        }
        country_code = match.group(1)
        return code_map.get(country_code, country_code)
    
    # If no country code, try ROR API
    return lookup_institute_country(affiliation)

def main():
    """
    Main function to process and organize participant data.
    """
    print("STEP 1: PROCESSING PARTICIPANT DATA")
    print("===================================")
    print("This script processes participant data from manually saved text files.")
    print("The data will be used to enhance speaker identification in the next step.")
    
    # Process all data files
    all_participants = process_all_data_files()
    
    if not all_participants:
        print("No participant data processed. Exiting.")
        return
    
    # Create JSON and CSV files
    create_combined_file(all_participants)
    create_csv_files(all_participants)
    
    # Print summary
    print_participant_summary(all_participants)
    
    print("\nParticipant data preparation complete!")
    print("Now you can run 'python QM/generate_conference_data.py' to process the conference data.")

if __name__ == "__main__":
    main()