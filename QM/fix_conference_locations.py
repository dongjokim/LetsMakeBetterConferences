import os
import json
import re

# List of known QM conference locations
LOCATION_MAP = {
    '2011': 'Annecy, France',
    '2012': 'Washington DC, USA', 
    '2014': 'Darmstadt, Germany',
    '2015': 'Kobe, Japan',
    '2017': 'Chicago, USA',
    '2018': 'Venice, Italy',
    '2019': 'Wuhan, China',
    '2022': 'Krakow, Poland',
    '2023': 'Houston, USA',
    '2025': 'Frankfurt, Germany'
}

def fix_locations():
    """Fix the locations in all processed data files"""
    print("Fixing conference locations in processed data files...")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory does not exist")
        return
    
    # Get all processed data files
    processed_files = [f for f in os.listdir(data_dir) if f.startswith('QM') and f.endswith('_processed_data.json')]
    
    if not processed_files:
        print("No processed data files found")
        return
    
    for filename in processed_files:
        # Extract year from filename (QM2018_processed_data.json -> 2018)
        year_match = re.search(r'QM(\d{4})_', filename)
        if not year_match:
            print(f"Could not extract year from filename: {filename}")
            continue
            
        year = year_match.group(1)
        filepath = os.path.join(data_dir, filename)
        
        # Get the known location for this year
        location = LOCATION_MAP.get(year, None)
        
        if not location:
            print(f"No known location for QM{year}, skipping {filename}")
            continue
            
        print(f"Updating location for QM{year} to: {location}")
        
        # Read the file
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update the location in metadata
            if 'metadata' in data:
                data['metadata']['conference_location'] = location
                
                # Write the updated file
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Successfully updated {filename}")
            else:
                print(f"Error: No metadata found in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("Location fixing complete")

if __name__ == "__main__":
    fix_locations() 