import requests
import json
import os
from datetime import datetime

def fetch_and_save_conference_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Read conference IDs
    with open('listofQMindigo', 'r') as f:
        conferences = [line.strip().split() for line in f]
    
    for year, indico_id in conferences:
        print(f"Fetching QM{year} (ID: {indico_id})...")
        
        # Define output file
        output_file = f'data/QM{year}_data.json'
        
        # Skip if already downloaded
        if os.path.exists(output_file):
            print(f"Data for QM{year} already exists, skipping...")
            continue
        
        try:
            # Fetch from Indico API
            url = f"https://indico.cern.ch/export/event/{indico_id}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Add metadata
            data['metadata'] = {
                'year': year,
                'indico_id': indico_id,
                'download_date': datetime.now().isoformat()
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Successfully saved data for QM{year}")
            
        except Exception as e:
            print(f"Error fetching QM{year}: {e}")

if __name__ == "__main__":
    fetch_and_save_conference_data() 