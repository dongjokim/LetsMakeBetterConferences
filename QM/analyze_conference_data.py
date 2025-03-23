import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec
import csv
import re
import traceback
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# Only keep the font settings needed for plots
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 'large',
    'axes.labelsize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'figure.titlesize': 'x-large'
})

# Set font that supports CJK characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Conference locations for reference
CONFERENCE_LOCATIONS = {
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

def load_processed_data(base_dir='data/processed'):
    """Load processed conference data from CSV files"""
    conference_data = {}
    
    try:
        # Check if base directory exists
        if not os.path.exists(base_dir):
            print(f"Error: Processed data directory not found at {base_dir}")
            print("Please run generate_conference_data.py first")
            return None
            
        # Get all year directories
        year_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        for year in sorted(year_dirs):
            year_path = os.path.join(base_dir, year)
            year_data = {}
            
            # Load each talk type
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'all_talks']:
                file_path = os.path.join(year_path, f'{talk_type}.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    year_data[talk_type] = df.to_dict('records')
            
            # Load statistics if they exist
            stats_file = os.path.join(year_path, 'statistics.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    year_data.update(stats)
            
            conference_data[year] = year_data
            
        print(f"\nLoaded processed data for {len(conference_data)} conferences")
        return conference_data
        
    except Exception as e:
        print(f"Error loading processed data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_participant_data():
    """Load participant data from text files in data/html directory"""
    participant_data = {}
    
    # Check if the data/html directory exists
    html_dir = 'data/html'
    if not os.path.exists(html_dir):
        print(f"Directory {html_dir} not found")
        return participant_data
    
    # Look for man_YYYY.data files
    for filename in os.listdir(html_dir):
        if filename.startswith('man_') and filename.endswith('.data'):
            try:
                # Extract year from filename
                year_match = re.search(r'man_(\d{4})\.data', filename)
                if year_match:
                    year = year_match.group(1)
                    file_path = os.path.join(html_dir, filename)
                    
                    # Read the text file
                    participants = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        # Skip header line
                        if len(lines) > 0:
                            header = lines[0].strip().lower()
                            data_lines = lines[1:]
                            
                            # Determine format based on header
                            has_id = 'id' in header
                            first_name_first = 'first name' in header and header.find('first name') < header.find('last name')
                            has_country_column = 'country' in header
                            
                            print(f"File {filename} format: ID={has_id}, FirstNameFirst={first_name_first}, CountryColumn={has_country_column}")
                            
                            # Process each data line
                            for line in data_lines:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                parts = line.split('\t')
                                parts = [p.strip() for p in parts if p.strip()]
                                
                                if len(parts) < 2:  # Need at least name and affiliation
                                    continue
                                
                                # Extract data based on format
                                if has_id:
                                    # Format: ID, First Name, Last Name, Affiliation
                                    if len(parts) >= 4:
                                        first_name = parts[1]
                                        last_name = parts[2]
                                        affiliation = parts[3]
                                        country = ''
                                    else:
                                        continue
                                elif has_country_column:
                                    # Format: Last Name, First Name, Affiliation, Country
                                    if len(parts) >= 4:
                                        last_name = parts[0]
                                        first_name = parts[1]
                                        affiliation = parts[2]
                                        country = parts[3]
                                    else:
                                        continue
                                elif first_name_first:
                                    # Format: First Name, Last Name, Affiliation
                                    if len(parts) >= 3:
                                        first_name = parts[0]
                                        last_name = parts[1]
                                        affiliation = parts[2]
                                        country = ''
                                    else:
                                        continue
                                else:
                                    # Format: Last Name, First Name, Affiliation
                                    if len(parts) >= 3:
                                        last_name = parts[0]
                                        first_name = parts[1]
                                        affiliation = parts[2]
                                        country = ''
                                    else:
                                        continue
                                
                                # Extract country from affiliation if not provided
                                if not country and '(' in affiliation and ')' in affiliation:
                                    country_match = re.search(r'\(([A-Z]{2})\)', affiliation)
                                    if country_match:
                                        country = country_match.group(1)
                                        
                                        # Convert country codes to full names
                                        country_codes = {
                                            'US': 'United States',
                                            'UK': 'United Kingdom',
                                            'DE': 'Germany',
                                            'FR': 'France',
                                            'IT': 'Italy',
                                            'JP': 'Japan',
                                            'CN': 'China',
                                            'RU': 'Russia',
                                            'IN': 'India',
                                            'BR': 'Brazil',
                                            'CA': 'Canada',
                                            'AU': 'Australia',
                                            'CH': 'Switzerland',
                                            'NL': 'Netherlands',
                                            'ES': 'Spain',
                                            'SE': 'Sweden',
                                            'PL': 'Poland',
                                            'AT': 'Austria',
                                            'BE': 'Belgium',
                                            'DK': 'Denmark',
                                            'FI': 'Finland',
                                            'NO': 'Norway',
                                            'PT': 'Portugal',
                                            'GR': 'Greece',
                                            'IE': 'Ireland',
                                            'CZ': 'Czech Republic',
                                            'HU': 'Hungary',
                                            'TR': 'Turkey',
                                            'IL': 'Israel',
                                            'ZA': 'South Africa',
                                            'MX': 'Mexico',
                                            'KR': 'South Korea',
                                            'SG': 'Singapore',
                                            'MY': 'Malaysia',
                                            'TH': 'Thailand',
                                            'ID': 'Indonesia',
                                            'PH': 'Philippines',
                                            'VN': 'Vietnam',
                                            'AR': 'Argentina',
                                            'CL': 'Chile',
                                            'CO': 'Colombia',
                                            'PE': 'Peru',
                                            'VE': 'Venezuela',
                                            'UA': 'Ukraine',
                                            'BY': 'Belarus',
                                            'RO': 'Romania',
                                            'BG': 'Bulgaria',
                                            'RS': 'Serbia',
                                            'HR': 'Croatia',
                                            'SI': 'Slovenia',
                                            'SK': 'Slovakia',
                                            'EE': 'Estonia',
                                            'LV': 'Latvia',
                                            'LT': 'Lithuania'
                                        }
                                        
                                        if country in country_codes:
                                            country = country_codes[country]
                                
                                # Combine first and last name
                                name = f"{first_name} {last_name}".strip()
                                
                                participants.append({
                                    'name': name,
                                    'affiliation': affiliation,
                                    'country': country
                                })
                    
                    participant_data[year] = participants
                    print(f"Loaded {len(participants)} participants from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                import traceback
                traceback.print_exc()
    
    # If no data was loaded, create empty data for each year
    years = ['2011', '2012', '2014', '2015', '2017', '2018', '2019', '2022', '2023', '2025']
    for year in years:
        if year not in participant_data:
            participant_data[year] = []
            print(f"No participant data available for {year}")
    
    return participant_data

def update_speaker_info_from_participant_data(conference_data, participant_data):
    """Update speaker information from participant data"""
    # Create a mapping of names to participant data
    participant_map = {}
    
    # Check if we have any real participant data
    has_real_data = False
    for year, participants in participant_data.items():
        if participants:  # If the list is not empty
            has_real_data = True
            break
    
    if not has_real_data:
        print("No actual participant data available, creating from conference data...")
        
        # Create participant data from conference data
        for year, data in conference_data.items():
            participant_data[year] = []
            
            # Add all speakers from all talk types
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type in data:
                    for talk in data[talk_type]:
                        speaker = talk.get('Speaker', '')
                        institute = talk.get('Institute', '')
                        country = talk.get('Country', '')
                        
                        if speaker and (institute != 'Unknown' or country != 'Unknown'):
                            # Add to participant data
                            participant_data[year].append({
                                'name': speaker,
                                'affiliation': institute,
                                'country': country
                            })
            
            print(f"Created {len(participant_data[year])} participant entries for year {year}")
    
    # Now build the mapping from all participant data
    for year, participants in participant_data.items():
        print(f"  Processing participant data for {year}...")
        for participant in participants:
            name = participant.get('name', '')
            if name:
                # Store both affiliation and country
                affiliation = participant.get('affiliation', '')
                country = participant.get('country', '')
                
                # Only store if we have valid data
                if affiliation or country:
                    if name not in participant_map:
                        participant_map[name] = {'Institute': '', 'Country': ''}
                    
                    # Update with better data if available
                    if affiliation and not participant_map[name]['Institute']:
                        participant_map[name]['Institute'] = affiliation
                    
                    if country and not participant_map[name]['Country']:
                        participant_map[name]['Country'] = country
    
    print(f"Created mapping for {len(participant_map)} unique participants")
    
    # Update speaker information in conference data
    updated_institute_count = 0
    updated_country_count = 0
    
    for year, data in conference_data.items():
        for talk_type in ['all_talks', 'plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    speaker = talk.get('Speaker', '')
                    
                    # Try exact match first
                    if speaker in participant_map:
                        # Update institute if unknown
                        if talk.get('Institute', '') == 'Unknown' and participant_map[speaker]['Institute']:
                            talk['Institute'] = participant_map[speaker]['Institute']
                            updated_institute_count += 1
                        
                        # Update country if unknown
                        if talk.get('Country', '') == 'Unknown' and participant_map[speaker]['Country']:
                            talk['Country'] = participant_map[speaker]['Country']
                            updated_country_count += 1
                    else:
                        # Try fuzzy matching for names
                        for participant_name in participant_map:
                            # Simple fuzzy match: check if one name is contained in the other
                            if (speaker in participant_name or participant_name in speaker) and len(speaker) > 5:
                                # Update institute if unknown
                                if talk.get('Institute', '') == 'Unknown' and participant_map[participant_name]['Institute']:
                                    talk['Institute'] = participant_map[participant_name]['Institute']
                                    updated_institute_count += 1
                                    print(f"Fuzzy matched: '{speaker}' with '{participant_name}'")
                                
                                # Update country if unknown
                                if talk.get('Country', '') == 'Unknown' and participant_map[participant_name]['Country']:
                                    talk['Country'] = participant_map[participant_name]['Country']
                                    updated_country_count += 1
                                
                                break
    
    print(f"Updated institute information for {updated_institute_count} talks")
    print(f"Updated country information for {updated_country_count} talks")
    return conference_data

def clean_institute_country_mappings(input_filename='unknown_institute_mappings.csv', 
                                    output_filename='cleaned_institute_mappings.csv'):
    """Clean and standardize institute-country mappings"""
    mappings = {}
    
    # Try to load existing mappings
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    institute = row[0].strip()
                    country = row[1].strip() if len(row) > 1 and row[1].strip() else 'Unknown'
                    mappings[institute] = country
        print(f"Loaded {len(mappings)} institute-country mappings from {input_filename}")
    except FileNotFoundError:
        print(f"Mapping file {input_filename} not found, creating new mappings")
    
    # Save cleaned mappings
    try:
        with open(output_filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Institute', 'Country'])
            for institute, country in sorted(mappings.items()):
                writer.writerow([institute, country])
        print(f"Saved {len(mappings)} cleaned mappings to {output_filename}")
    except Exception as e:
        print(f"Error saving cleaned mappings: {e}")
    
    return mappings

def fix_unknown_institute_country_data(conference_data):
    """Fix unknown institute and country data using mappings"""
    # Load institute-country mappings
    try:
        mappings = {}
        with open('cleaned_institute_mappings.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    institute = row[0].strip()
                    country = row[1].strip()
                    mappings[institute] = country
        print(f"Loaded {len(mappings)} institute-country mappings")
    except FileNotFoundError:
        print("Mapping file not found, no fixes will be applied")
        return conference_data
    
    # Apply fixes
    fixed_count = 0
    
    for year, data in conference_data.items():
        for talk_type in ['all_talks', 'plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    institute = talk.get('Institute', '')
                    if institute in mappings and talk.get('Country', '') == 'Unknown':
                        talk['Country'] = mappings[institute]
                        fixed_count += 1
    
    print(f"Fixed {fixed_count} unknown country entries")
    return conference_data

def fix_common_affiliation_problems(conference_data):
    """Fix common affiliation problems"""
    # Define common fixes
    COMMON_FIXES = {
        # USA institutions
        'BNL': {'Institute': 'Brookhaven National Laboratory', 'Country': 'USA'},
        'LBNL': {'Institute': 'Lawrence Berkeley National Laboratory', 'Country': 'USA'},
        'ORNL': {'Institute': 'Oak Ridge National Laboratory', 'Country': 'USA'},
        'LANL': {'Institute': 'Los Alamos National Laboratory', 'Country': 'USA'},
        'ANL': {'Institute': 'Argonne National Laboratory', 'Country': 'USA'},
        'FNAL': {'Institute': 'Fermi National Accelerator Laboratory', 'Country': 'USA'},
        'JLab': {'Institute': 'Jefferson Laboratory', 'Country': 'USA'},
        'MIT': {'Institute': 'Massachusetts Institute of Technology', 'Country': 'USA'},
        
        # European institutions
        'CERN': {'Institute': 'CERN', 'Country': 'Switzerland'},
        'GSI': {'Institute': 'GSI Helmholtz Centre', 'Country': 'Germany'},
        'DESY': {'Institute': 'DESY', 'Country': 'Germany'},
        'JINR': {'Institute': 'Joint Institute for Nuclear Research', 'Country': 'Russia'},
        
        # Asian institutions
        'RIKEN': {'Institute': 'RIKEN', 'Country': 'Japan'},
        'KEK': {'Institute': 'KEK', 'Country': 'Japan'},
        'TIFR': {'Institute': 'Tata Institute of Fundamental Research', 'Country': 'India'},
        'VECC': {'Institute': 'Variable Energy Cyclotron Centre', 'Country': 'India'}
    }
    
    # Apply fixes
    fixed_count = 0
    
    for year, data in conference_data.items():
        for talk_type in ['all_talks', 'plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    institute = talk.get('Institute', '')
                    
                    # Check for exact matches
                    if institute in COMMON_FIXES:
                        talk['Institute'] = COMMON_FIXES[institute]['Institute']
                        talk['Country'] = COMMON_FIXES[institute]['Country']
                        fixed_count += 1
                        continue
                    
                    # Check for partial matches
                    for key, fix in COMMON_FIXES.items():
                        if key in institute and talk.get('Country', '') == 'Unknown':
                            talk['Country'] = fix['Country']
                            fixed_count += 1
                            break
    
    print(f"Fixed {fixed_count} common affiliation problems")
    return conference_data

def add_manual_country_fixes(conference_data):
    """Add manual fixes for country information"""
    # Define manual fixes for specific institutions
    MANUAL_FIXES = {
        # USA institutions
        'Lawrence Berkeley National Laboratory': 'USA',
        'Berkeley Lab': 'USA',
        'Lawrence Livermore National Laboratory': 'USA',
        'Oak Ridge National Laboratory': 'USA',
        'Los Alamos National Laboratory': 'USA',
        'Argonne National Laboratory': 'USA',
        'Brookhaven National Laboratory': 'USA',
        'Fermi National Accelerator Laboratory': 'USA',
        'Fermilab': 'USA',
        'Jefferson Lab': 'USA',
        'SLAC National Accelerator Laboratory': 'USA',
        'Pacific Northwest National Laboratory': 'USA',
        'Sandia National Laboratories': 'USA',
        'Idaho National Laboratory': 'USA',
        'Ames Laboratory': 'USA',
        'Princeton Plasma Physics Laboratory': 'USA',
        'National Renewable Energy Laboratory': 'USA',
        'National Energy Technology Laboratory': 'USA',
        'National Institute of Standards and Technology': 'USA',
        'NASA': 'USA',
        'Jet Propulsion Laboratory': 'USA',
        'Massachusetts Institute of Technology': 'USA',
        'MIT': 'USA',
        'Harvard University': 'USA',
        'Stanford University': 'USA',
        'California Institute of Technology': 'USA',
        'Caltech': 'USA',
        'University of California': 'USA',
        'UC Berkeley': 'USA',
        'UC Davis': 'USA',
        'UC Irvine': 'USA',
        'UC Los Angeles': 'USA',
        'UCLA': 'USA',
        'UC Riverside': 'USA',
        'UC San Diego': 'USA',
        'UCSD': 'USA',
        'UC Santa Barbara': 'USA',
        'UCSB': 'USA',
        'UC Santa Cruz': 'USA',
        'UCSC': 'USA',
        'UC San Francisco': 'USA',
        'UCSF': 'USA',
        'UC Merced': 'USA',
        'University of Chicago': 'USA',
        'University of Michigan': 'USA',
        'University of Wisconsin': 'USA',
        'University of Illinois': 'USA',
        'University of Texas': 'USA',
        'University of Washington': 'USA',
        'University of Colorado': 'USA',
        'University of Minnesota': 'USA',
        'University of Pennsylvania': 'USA',
        'University of Florida': 'USA',
        'University of North Carolina': 'USA',
        'University of Virginia': 'USA',
        'University of Maryland': 'USA',
        'University of Pittsburgh': 'USA',
        'University of Arizona': 'USA',
        'University of Rochester': 'USA',
        'University of Utah': 'USA',
        'University of Iowa': 'USA',
        'University of Oregon': 'USA',
        'University of Kansas': 'USA',
        'University of Kentucky': 'USA',
        'University of Tennessee': 'USA',
        'University of Alabama': 'USA',
        'University of Georgia': 'USA',
        'University of South Carolina': 'USA',
        'University of Oklahoma': 'USA',
        'University of Nebraska': 'USA',
        'University of Missouri': 'USA',
        'University of Arkansas': 'USA',
        'University of Mississippi': 'USA',
        'University of Louisiana': 'USA',
        'University of Hawaii': 'USA',
        'University of Alaska': 'USA',
        'University of Delaware': 'USA',
        'University of Rhode Island': 'USA',
        'University of Vermont': 'USA',
        'University of New Hampshire': 'USA',
        'University of Maine': 'USA',
        'University of Connecticut': 'USA',
        'University of Massachusetts': 'USA',
        'University of New Mexico': 'USA',
        'University of Nevada': 'USA',
        'University of Idaho': 'USA',
        'University of Montana': 'USA',
        'University of Wyoming': 'USA',
        'University of North Dakota': 'USA',
        'University of South Dakota': 'USA',
        'Yale University': 'USA',
        'Princeton University': 'USA',
        'Columbia University': 'USA',
        'Cornell University': 'USA',
        'Brown University': 'USA',
        'Dartmouth College': 'USA',
        'University of Pennsylvania': 'USA',
        'Duke University': 'USA',
        'Johns Hopkins University': 'USA',
        'Northwestern University': 'USA',
        'Vanderbilt University': 'USA',
        'Rice University': 'USA',
        'Emory University': 'USA',
        'Washington University in St. Louis': 'USA',
        'University of Notre Dame': 'USA',
        'Georgetown University': 'USA',
        'Carnegie Mellon University': 'USA',
        'University of Southern California': 'USA',
        'New York University': 'USA',
        'Boston University': 'USA',
        'Tufts University': 'USA',
        'Case Western Reserve University': 'USA',
        'University of Rochester': 'USA',
        'Brandeis University': 'USA',
        'College of William & Mary': 'USA',
        'University of Miami': 'USA',
        'Northeastern University': 'USA',
        'Rensselaer Polytechnic Institute': 'USA',
        'University of California, Berkeley': 'USA',
        'University of California, Los Angeles': 'USA',
        'University of California, San Diego': 'USA',
        'University of California, Santa Barbara': 'USA',
        'University of California, Irvine': 'USA',
        'University of California, Davis': 'USA',
        'University of California, Santa Cruz': 'USA',
        'University of California, Riverside': 'USA',
        'University of California, Merced': 'USA',
        'University of California, San Francisco': 'USA',
        'University of Texas at Austin': 'USA',
        'University of Texas at Dallas': 'USA',
        'University of Texas at Arlington': 'USA',
        'University of Texas at San Antonio': 'USA',
        'University of Texas at El Paso': 'USA',
        'University of Texas Rio Grande Valley': 'USA',
        'University of Texas Medical Branch': 'USA',
        'University of Texas Southwestern Medical Center': 'USA',
        'University of Texas Health Science Center': 'USA',
        'University of Texas MD Anderson Cancer Center': 'USA',
        'University of Illinois at Urbana-Champaign': 'USA',
        'University of Illinois at Chicago': 'USA',
        'University of Illinois at Springfield': 'USA',
        'University of Michigan-Ann Arbor': 'USA',
        'University of Michigan-Dearborn': 'USA',
        'University of Michigan-Flint': 'USA',
        'University of Wisconsin-Madison': 'USA',
        'University of Wisconsin-Milwaukee': 'USA',
        'University of Wisconsin-Green Bay': 'USA',
        'University of Wisconsin-La Crosse': 'USA',
        'University of Wisconsin-Eau Claire': 'USA',
        'University of Wisconsin-Oshkosh': 'USA',
        'University of Wisconsin-Whitewater': 'USA',
        'University of Wisconsin-Stout': 'USA',
        'University of Wisconsin-Stevens Point': 'USA',
        'University of Wisconsin-Platteville': 'USA',
        'University of Wisconsin-River Falls': 'USA',
        'University of Wisconsin-Superior': 'USA',
        'University of Wisconsin-Parkside': 'USA',
        'University of Minnesota-Twin Cities': 'USA',
        'University of Minnesota-Duluth': 'USA',
        'University of Minnesota-Morris': 'USA',
        'University of Minnesota-Crookston': 'USA',
        'University of Minnesota-Rochester': 'USA',
        
        # European institutions
        'CERN': 'Switzerland',
        'European Organization for Nuclear Research': 'Switzerland',
        'Paul Scherrer Institute': 'Switzerland',
        'ETH Zurich': 'Switzerland',
        'EPFL': 'Switzerland',
        'University of Zurich': 'Switzerland',
        'University of Geneva': 'Switzerland',
        'University of Bern': 'Switzerland',
        'University of Basel': 'Switzerland',
        'University of Lausanne': 'Switzerland',
        'University of Fribourg': 'Switzerland',
        'University of Neuchâtel': 'Switzerland',
        'University of St. Gallen': 'Switzerland',
        'University of Lucerne': 'Switzerland',
        'University of Lugano': 'Switzerland',
        
        'GSI Helmholtz Centre for Heavy Ion Research': 'Germany',
        'GSI': 'Germany',
        'DESY': 'Germany',
        'Deutsches Elektronen-Synchrotron': 'Germany',
        'Max Planck Institute': 'Germany',
        'Helmholtz Association': 'Germany',
        'Fraunhofer Society': 'Germany',
        'Leibniz Association': 'Germany',
        'German Research Foundation': 'Germany',
        'Technical University of Munich': 'Germany',
        'Ludwig Maximilian University of Munich': 'Germany',
        'Heidelberg University': 'Germany',
        'RWTH Aachen University': 'Germany',
        'Humboldt University of Berlin': 'Germany',
        'Free University of Berlin': 'Germany',
        'Technical University of Berlin': 'Germany',
        'University of Göttingen': 'Germany',
        'University of Bonn': 'Germany',
        'University of Cologne': 'Germany',
        'University of Frankfurt': 'Germany',
        'University of Hamburg': 'Germany',
        'University of Münster': 'Germany',
        'University of Tübingen': 'Germany',
        'University of Freiburg': 'Germany',
        'University of Erlangen-Nuremberg': 'Germany',
        'University of Würzburg': 'Germany',
        'University of Stuttgart': 'Germany',
        'University of Mannheim': 'Germany',
        'University of Konstanz': 'Germany',
        'University of Jena': 'Germany',
        'University of Kiel': 'Germany',
        'University of Regensburg': 'Germany',
        'University of Mainz': 'Germany',
        'University of Marburg': 'Germany',
        'University of Giessen': 'Germany',
        'University of Bielefeld': 'Germany',
        'University of Bochum': 'Germany',
        'University of Dortmund': 'Germany',
        'University of Duisburg-Essen': 'Germany',
        'University of Düsseldorf': 'Germany',
        'University of Hannover': 'Germany',
        'University of Bremen': 'Germany',
        'University of Rostock': 'Germany',
        'University of Greifswald': 'Germany',
        'University of Magdeburg': 'Germany',
        'University of Halle': 'Germany',
        'University of Leipzig': 'Germany',
        'University of Dresden': 'Germany',
        'University of Chemnitz': 'Germany',
        'University of Bayreuth': 'Germany',
        'University of Passau': 'Germany',
        'University of Augsburg': 'Germany',
        'University of Ulm': 'Germany',
        'University of Hohenheim': 'Germany',
        'University of Kaiserslautern': 'Germany',
        'University of Saarland': 'Germany',
        'University of Trier': 'Germany',
        'University of Koblenz-Landau': 'Germany',
        'University of Siegen': 'Germany',
        'University of Paderborn': 'Germany',
        'University of Wuppertal': 'Germany',
        'University of Oldenburg': 'Germany',
        'University of Osnabrück': 'Germany',
        'University of Lüneburg': 'Germany',
        'University of Hildesheim': 'Germany',
        'University of Vechta': 'Germany',
        'University of Flensburg': 'Germany',
        'University of Bamberg': 'Germany',
        'University of Eichstätt-Ingolstadt': 'Germany',
        'University of Erfurt': 'Germany',
        'University of Weimar': 'Germany',
        'University of Ilmenau': 'Germany',
        'University of Potsdam': 'Germany',
        'University of Frankfurt (Oder)': 'Germany',
        'University of Cottbus': 'Germany',
        'University of the Federal Armed Forces Munich': 'Germany',
        'University of the Federal Armed Forces Hamburg': 'Germany',
        'University of Applied Sciences': 'Germany',
        'Karlsruhe Institute of Technology': 'Germany',
        'KIT': 'Germany',
        
        'JINR': 'Russia',
        'Joint Institute for Nuclear Research': 'Russia',
        'Moscow State University': 'Russia',
        'Lomonosov Moscow State University': 'Russia',
        'Saint Petersburg State University': 'Russia',
        'Novosibirsk State University': 'Russia',
        'Moscow Institute of Physics and Technology': 'Russia',
        'MIPT': 'Russia',
        'National Research Nuclear University MEPhI': 'Russia',
        'MEPhI': 'Russia',
        'Moscow Engineering Physics Institute': 'Russia',
        'Kurchatov Institute': 'Russia',
        'Institute for Theoretical and Experimental Physics': 'Russia',
        'ITEP': 'Russia',
        'Institute for High Energy Physics': 'Russia',
        'IHEP': 'Russia',
        'Budker Institute of Nuclear Physics': 'Russia',
        'BINP': 'Russia',
        'Petersburg Nuclear Physics Institute': 'Russia',
        'PNPI': 'Russia',
        'Russian Academy of Sciences': 'Russia',
        'RAS': 'Russia',
        
        # Asian institutions
        'RIKEN': 'Japan',
        'KEK': 'Japan',
        'High Energy Accelerator Research Organization': 'Japan',
        'Japan Atomic Energy Agency': 'Japan',
        'JAEA': 'Japan',
        'University of Tokyo': 'Japan',
        'Kyoto University': 'Japan',
        'Osaka University': 'Japan',
        'Tohoku University': 'Japan',
        'Nagoya University': 'Japan',
        'Kyushu University': 'Japan',
        'Hokkaido University': 'Japan',
        'Tokyo Institute of Technology': 'Japan',
        'Waseda University': 'Japan',
        'Keio University': 'Japan',
        'Tsukuba University': 'Japan',
        'University of Tsukuba': 'Japan',
        'Hiroshima University': 'Japan',
        'Kobe University': 'Japan',
        'Okayama University': 'Japan',
        'Kanazawa University': 'Japan',
        'Chiba University': 'Japan',
        'Niigata University': 'Japan',
        'Shinshu University': 'Japan',
        'Yamagata University': 'Japan',
        'Yamaguchi University': 'Japan',
        'Yamanashi University': 'Japan',
        'Ehime University': 'Japan',
        'Kagoshima University': 'Japan',
        'Ibaraki University': 'Japan',
        'Gunma University': 'Japan',
        'Saitama University': 'Japan',
        'Tottori University': 'Japan',
        'Shimane University': 'Japan',
        'Tokushima University': 'Japan',
        'Kagawa University': 'Japan',
        'Kochi University': 'Japan',
        'Saga University': 'Japan',
        'Oita University': 'Japan',
        'Miyazaki University': 'Japan',
        'Kumamoto University': 'Japan',
        'Fukuoka University': 'Japan',
        'Fukushima University': 'Japan',
        'Fukui University': 'Japan',
        'Toyama University': 'Japan',
        'Gifu University': 'Japan',
        'Mie University': 'Japan',
        'Shiga University': 'Japan',
        'Wakayama University': 'Japan',
        'Hyogo University': 'Japan',
        'Aichi University': 'Japan',
        'Ishikawa University': 'Japan',
        'Iwate University': 'Japan',
        'Akita University': 'Japan',
        'Miyagi University': 'Japan',
        'Aomori University': 'Japan',
        
        'Tsinghua University': 'China',
        'Peking University': 'China',
        'University of Science and Technology of China': 'China',
        'USTC': 'China',
        'Fudan University': 'China',
        'Shanghai Jiao Tong University': 'China',
        'Zhejiang University': 'China',
        'Nanjing University': 'China',
        'Wuhan University': 'China',
        'Huazhong University of Science and Technology': 'China',
        'Sun Yat-sen University': 'China',
        'Harbin Institute of Technology': 'China',
        'Beijing Normal University': 'China',
        'Nankai University': 'China',
        'Tianjin University': 'China',
        'Xiamen University': 'China',
        'Shandong University': 'China',
        'Sichuan University': 'China',
        'Jilin University': 'China',
        'Lanzhou University': 'China',
        'Northeastern University': 'China',
        'Central South University': 'China',
        'Southeast University': 'China',
        'South China University of Technology': 'China',
        'Hunan University': 'China',
        'Chongqing University': 'China',
        'Beijing Institute of Technology': 'China',
        'University of Electronic Science and Technology of China': 'China',
        'East China Normal University': 'China',
        'Chinese Academy of Sciences': 'China',
        'CAS': 'China',
        'Institute of High Energy Physics': 'China',
        'IHEP': 'China',
        'Institute of Modern Physics': 'China',
        'IMP': 'China',
        'Institute of Theoretical Physics': 'China',
        'ITP': 'China',
        'Institute of Physics': 'China',
        'IoP': 'China',
        'Shanghai Institute of Applied Physics': 'China',
        'SINAP': 'China',
        'China Institute of Atomic Energy': 'China',
        'CIAE': 'China',
        
        'TIFR': 'India',
        'Tata Institute of Fundamental Research': 'India',
        'VECC': 'India',
        'Variable Energy Cyclotron Centre': 'India',
        'Bhabha Atomic Research Centre': 'India',
        'BARC': 'India',
        'Indian Institute of Science': 'India',
        'IISc': 'India',
        'Indian Institute of Technology': 'India',
        'IIT Bombay': 'India',
        'IIT Delhi': 'India',
        'IIT Kanpur': 'India',
        'IIT Kharagpur': 'India',
        'IIT Madras': 'India',
        'IIT Roorkee': 'India',
        'IIT Guwahati': 'India',
        'IIT Hyderabad': 'India',
        'IIT Gandhinagar': 'India',
        'IIT Patna': 'India',
        'IIT Bhubaneswar': 'India',
        'IIT Indore': 'India',
        'IIT Mandi': 'India',
        'IIT Jodhpur': 'India',
        'IIT Ropar': 'India',
        'IIT Tirupati': 'India',
        'IIT Palakkad': 'India',
        'IIT Jammu': 'India',
        'IIT Dharwad': 'India',
        'IIT Bhilai': 'India',
        'IIT Goa': 'India',
        'Indian Institute of Science Education and Research': 'India',
        'IISER Pune': 'India',
        'IISER Kolkata': 'India',
        'IISER Mohali': 'India',
        'IISER Bhopal': 'India',
        'IISER Thiruvananthapuram': 'India',
        'IISER Tirupati': 'India',
        'IISER Berhampur': 'India',
        'National Institute of Science Education and Research': 'India',
        'NISER': 'India',
        'Saha Institute of Nuclear Physics': 'India',
        'SINP': 'India',
        'Institute of Physics': 'India',
        'IoP': 'India',
        'Harish-Chandra Research Institute': 'India',
        'HRI': 'India',
        'Institute of Mathematical Sciences': 'India',
        'IMSc': 'India',
        'Inter-University Centre for Astronomy and Astrophysics': 'India',
        'IUCAA': 'India',
        'Jawaharlal Nehru University': 'India',
        'JNU': 'India',
        'University of Delhi': 'India',
        'DU': 'India',
        'Banaras Hindu University': 'India',
        'BHU': 'India',
        'Aligarh Muslim University': 'India',
        'AMU': 'India',
        'Jadavpur University': 'India',
        'University of Calcutta': 'India',
        'University of Mumbai': 'India',
        'University of Madras': 'India',
        'University of Hyderabad': 'India',
        'Panjab University': 'India',
        'Savitribai Phule Pune University': 'India',
        'Anna University': 'India',
        'Jamia Millia Islamia': 'India',
        'Jamia': 'India',
        'Visva-Bharati University': 'India',
        'Bhabha Atomic Research Centre': 'India',
        'BARC': 'India',
        'Raja Ramanna Centre for Advanced Technology': 'India',
        'RRCAT': 'India',
        'Indira Gandhi Centre for Atomic Research': 'India',
        'IGCAR': 'India',
        'Institute for Plasma Research': 'India',
        'IPR': 'India',
        'Physical Research Laboratory': 'India',
        'PRL': 'India',
        'National Physical Laboratory': 'India',
        'NPL': 'India',
        'Indian Association for the Cultivation of Science': 'India',
        'IACS': 'India',
        'S.N. Bose National Centre for Basic Sciences': 'India',
        'SNBNCBS': 'India',
        'Raman Research Institute': 'India',
        'RRI': 'India',
        'Indian Space Research Organisation': 'India',
        'ISRO': 'India',
        'Defence Research and Development Organisation': 'India',
        'DRDO': 'India',
        'Council of Scientific and Industrial Research': 'India',
        'CSIR': 'India',
        'Department of Atomic Energy': 'India',
        'DAE': 'India',
        'Department of Science and Technology': 'India',
        'DST': 'India'
    }
    
    # Apply manual fixes
    fixed_count = 0
    
    for year, data in conference_data.items():
        for talk_type in ['all_talks', 'plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    institute = talk.get('Institute', '')
                    
                    # Check for exact matches
                    if institute in MANUAL_FIXES and talk.get('Country', '') == 'Unknown':
                        talk['Country'] = MANUAL_FIXES[institute]
                        fixed_count += 1
                        continue
                    
                    # Check for partial matches
                    for inst, country in MANUAL_FIXES.items():
                        if inst in institute and talk.get('Country', '') == 'Unknown':
                            talk['Country'] = country
                            fixed_count += 1
                            break
    
    print(f"Manually fixed {fixed_count} country entries")
    return conference_data

def debug_conference_data(data, label="Conference Data"):
    """
    Debug function to print information about the conference data structure.
    
    Parameters:
    - data: The data to debug
    - label: A label for the debug output
    """
    print(f"\n=== DEBUG: {label} ===")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        print(f"Sample value type: {type(next(iter(data.values()))) if data else 'No values'}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"First few items: {data[:3] if data else 'Empty list'}")
    else:
        print(f"Value: {data}")
    
    print("=" * 50)

def filter_relevant_talk_types(conference_data):
    """
    Filter conference data to only include plenary, parallel, and poster talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Updated conference_data with only relevant talk types
    """
    print("\nFiltering data to include only plenary, parallel, and poster talks...")
    
    for year, data in conference_data.items():
        # Create new all_talks list with only the relevant talk types
        all_talks = []
        
        # Add all plenary talks
        if 'plenary_talks' in data:
            all_talks.extend(data['plenary_talks'])
        
        # Add all parallel talks
        if 'parallel_talks' in data:
            all_talks.extend(data['parallel_talks'])
        
        # Add all poster talks
        if 'poster_talks' in data:
            all_talks.extend(data['poster_talks'])
        
        # Update the all_talks entry
        data['all_talks'] = all_talks
        
        # Remove other talk types if they exist
        if 'flash_talks' in data:
            print(f"  Removed {len(data['flash_talks'])} flash talks from QM{year}")
            del data['flash_talks']
        
        if 'other_talks' in data:
            print(f"  Removed {len(data['other_talks'])} other talks from QM{year}")
            del data['other_talks']
    
    return conference_data

def fix_unknown_plenary_parallel_talks(conference_data):
    """
    Specifically target and fix unknown plenary and parallel talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Updated conference_data with fewer unknown plenary and parallel talks
    """
    print("\nSpecifically fixing unknown plenary and parallel talks...")
    
    # Create a mapping of speakers to their known affiliations and countries
    speaker_mapping = {}
    
    # First pass: collect all known speaker information across all conferences
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    speaker = talk.get('Speaker', '')
                    institute = talk.get('Institute', '')
                    country = talk.get('Country', '')
                    
                    if speaker and institute != 'Unknown' and country != 'Unknown':
                        speaker_mapping[speaker] = {
                            'Institute': institute,
                            'Country': country
                        }
    
    print(f"Created mapping for {len(speaker_mapping)} speakers with known affiliations")
    
    # Second pass: apply the collected information to unknown entries
    fixed_plenary = 0
    fixed_parallel = 0
    
    for year, data in conference_data.items():
        # Fix plenary talks
        if 'plenary_talks' in data:
            for talk in data['plenary_talks']:
                speaker = talk.get('Speaker', '')
                if speaker in speaker_mapping and talk.get('Country', '') == 'Unknown':
                    talk['Institute'] = speaker_mapping[speaker]['Institute']
                    talk['Country'] = speaker_mapping[speaker]['Country']
                    fixed_plenary += 1
        
        # Fix parallel talks
        if 'parallel_talks' in data:
            for talk in data['parallel_talks']:
                speaker = talk.get('Speaker', '')
                if speaker in speaker_mapping and talk.get('Country', '') == 'Unknown':
                    talk['Institute'] = speaker_mapping[speaker]['Institute']
                    talk['Country'] = speaker_mapping[speaker]['Country']
                    fixed_parallel += 1
    
    print(f"Fixed {fixed_plenary} unknown plenary talks")
    print(f"Fixed {fixed_parallel} unknown parallel talks")
    
    return conference_data

def display_conference_summary(conference_data):
    """
    Display a summary of the conference data.
    
    Parameters:
    - conference_data: Dictionary with conference data
    """
    print("\nYear Location                  Total  Plenary  Parallel  Poster  Flash  Unk_Plen Unk_Par")
    print("-" * 85)
    
    # Sort years
    years = sorted(conference_data.keys())
    
    for year in years:
        data = conference_data[year]
        location = CONFERENCE_LOCATIONS.get(year, 'Unknown location')
        total = len(data.get('all_talks', []))
        plenary = len(data.get('plenary_talks', []))
        parallel = len(data.get('parallel_talks', []))
        poster = len(data.get('poster_talks', []))
        flash = len(data.get('flash_talks', [])) if 'flash_talks' in data else 0
        
        # Count unknown institutes - specifically looking at Institute field
        unknown_plenary = sum(1 for t in data.get('plenary_talks', []) if t.get('Institute', '') == 'Unknown')
        unknown_parallel = sum(1 for t in data.get('parallel_talks', []) if t.get('Institute', '') == 'Unknown')
        
        print(f"{year} {location:<25} {total:<6} {plenary:<8} {parallel:<8} {poster:<6} {flash:<5} {unknown_plenary:<8} {unknown_parallel}")

def fix_unknown_institutes(conference_data):
    """
    Specifically target and fix unknown institutes for plenary and parallel talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Updated conference_data with fewer unknown institutes
    """
    print("\nSpecifically fixing unknown institutes for plenary and parallel talks...")
    
    # Create a mapping of speakers to their known institutes
    speaker_institute_mapping = {}
    
    # First pass: collect all known speaker institute information across all conferences
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    speaker = talk.get('Speaker', '')
                    institute = talk.get('Institute', '')
                    
                    if speaker and institute != 'Unknown':
                        speaker_institute_mapping[speaker] = institute
    
    print(f"Created mapping for {len(speaker_institute_mapping)} speakers with known institutes")
    
    # Second pass: apply the collected information to unknown entries
    fixed_plenary = 0
    fixed_parallel = 0
    
    for year, data in conference_data.items():
        # Fix plenary talks
        if 'plenary_talks' in data:
            for talk in data['plenary_talks']:
                speaker = talk.get('Speaker', '')
                if speaker in speaker_institute_mapping and talk.get('Institute', '') == 'Unknown':
                    talk['Institute'] = speaker_institute_mapping[speaker]
                    fixed_plenary += 1
        
        # Fix parallel talks
        if 'parallel_talks' in data:
            for talk in data['parallel_talks']:
                speaker = talk.get('Speaker', '')
                if speaker in speaker_institute_mapping and talk.get('Institute', '') == 'Unknown':
                    talk['Institute'] = speaker_institute_mapping[speaker]
                    fixed_parallel += 1
    
    print(f"Fixed {fixed_plenary} unknown institutes for plenary talks")
    print(f"Fixed {fixed_parallel} unknown institutes for parallel talks")
    
    return conference_data

def analyze_country_diversity(conference_data):
    """Analyze the diversity of countries in the conference data over time"""
    print("\nAnalyzing country diversity...")
    
    # Get years in sorted order
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Track country counts by year and overall
    country_by_year = {}
    unique_countries_by_year = {}
    hhi_by_year = {}
    all_time_counts = Counter()
    
    # Process each year
    for year in years:
        country_counts = Counter()
        
        # Count countries for all talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type not in conference_data[year]:
                continue
                
            for talk in conference_data[year][talk_type]:
                country = talk.get('Country', 'Unknown')
                if country and country != 'Unknown':
                    country_counts[country] += 1
        
        # Store counts and calculate metrics for this year
        country_by_year[year] = country_counts
        unique_countries_by_year[year] = len(country_counts)
        
        # Calculate HHI (Herfindahl-Hirschman Index) for concentration
        total_talks = sum(country_counts.values())
        if total_talks > 0:
            hhi = sum((count/total_talks)**2 for count in country_counts.values())
            hhi_by_year[year] = hhi * 10000  # Scale to traditional 0-10000 range
        else:
            hhi_by_year[year] = 0
    
    # Calculate all-time country stats
    for year_counts in country_by_year.values():
        all_time_counts.update(year_counts)
    
    # Print summary table
    print("\nTop countries by year (number of presentations):")
    top_n = 10
    top_countries_all_time = [country for country, _ in all_time_counts.most_common(top_n)]
    
    print(f"{'Country':<15} " + " ".join(f"{year:>7}" for year in years))
    print("-" * (15 + 8 * len(years)))
    
    for country in top_countries_all_time:
        row = [country]
        for year in years:
            row.append(country_by_year[year].get(country, 0))
        
        print(f"{country:<15} " + " ".join(f"{count:>7}" for count in row[1:]))
    
    return unique_countries_by_year, hhi_by_year

def create_diversity_metrics_plot(years, unique_countries_by_year, hhi_by_year):
    """Create plot showing diversity metrics over time"""
    print("Creating diversity metrics visualization...")
    
    # Extract data in the correct order
    unique_counts = [unique_countries_by_year.get(year, 0) for year in years]
    hhi_values = [hhi_by_year.get(year, 0) for year in years]
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # First y-axis: Unique countries
    color = 'tab:blue'
    ax1.set_xlabel('Conference Year')
    ax1.set_ylabel('Number of Unique Countries', color=color)
    ax1.bar(years, unique_counts, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Annotate values
    for i, count in enumerate(unique_counts):
        ax1.annotate(f"{count}", (years[i], count), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center',
                   color=color)
    
    # Second y-axis: HHI
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Herfindahl-Hirschman Index (HHI)', color=color)
    ax2.plot(years, hhi_values, color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Annotate values
    for i, hhi in enumerate(hhi_values):
        ax2.annotate(f"{hhi:.0f}", (years[i], hhi), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center',
                   color=color)
    
    plt.title('Diversity Metrics Over Time')
    plt.tight_layout()
    plt.savefig('figures/diversity_metrics.pdf', bbox_inches='tight')
    plt.close()

def create_representation_ratio_plot(plenary_country, parallel_country):
    """Create a plot of representation ratios between plenary and parallel talks"""
    print("Creating representation ratio visualization...")
    
    # Get top countries by total talks
    countries = set(plenary_country.keys()) | set(parallel_country.keys())
    
    # Calculate representation ratios
    ratios = {}
    total_plenary = sum(plenary_country.values())
    total_parallel = sum(parallel_country.values())
    
    for country in countries:
        if country == 'Unknown':
            continue
        
        plenary_pct = plenary_country.get(country, 0) / total_plenary if total_plenary > 0 else 0
        parallel_pct = parallel_country.get(country, 0) / total_parallel if total_parallel > 0 else 0
        
        # Avoid division by zero
        if parallel_pct > 0:
            ratio = plenary_pct / parallel_pct
        else:
            ratio = 0 if plenary_pct == 0 else float('inf')
        
        ratios[country] = ratio
    
    # Filter countries with at least a minimum number of talks
    min_total_talks = 5
    filtered_ratios = {country: ratio for country, ratio in ratios.items() 
                     if plenary_country.get(country, 0) + parallel_country.get(country, 0) >= min_total_talks}
    
    # Sort countries by ratio and get top/bottom countries
    top_countries = sorted(filtered_ratios.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 20 countries by talk count for visualization
    country_talk_count = {c: plenary_country.get(c, 0) + parallel_country.get(c, 0) 
                      for c in filtered_ratios.keys()}
    viz_countries = [c for c, _ in sorted(country_talk_count.items(), 
                                      key=lambda x: x[1], reverse=True)[:20]]
    
    # Prepare data for plotting
    plot_countries = []
    ratio_values = []
    
    for country in viz_countries:
        plot_countries.append(country)
        ratio_values.append(filtered_ratios[country])
    
    # Define region mapping
    country_to_region = {
        'USA': 'North America', 'Canada': 'North America',
        'Germany': 'Europe', 'UK': 'Europe', 'France': 'Europe', 'Italy': 'Europe',
        'Switzerland': 'Europe', 'Netherlands': 'Europe', 'Spain': 'Europe',
        'Poland': 'Europe', 'Czech Republic': 'Europe', 'Hungary': 'Europe',
        'Austria': 'Europe', 'Belgium': 'Europe', 'Denmark': 'Europe',
        'Finland': 'Europe', 'Norway': 'Europe', 'Sweden': 'Europe',
        'Portugal': 'Europe', 'Greece': 'Europe', 'Russia': 'Europe',
        'Ukraine': 'Europe', 'Serbia': 'Europe', 'Croatia': 'Europe',
        'Romania': 'Europe', 'Bulgaria': 'Europe', 'Slovakia': 'Europe',
        'Japan': 'Asia', 'China': 'Asia', 'India': 'Asia', 'South Korea': 'Asia',
        'Israel': 'Asia', 'Taiwan': 'Asia', 'Singapore': 'Asia',
        'Brazil': 'South America', 'Mexico': 'South America', 'Argentina': 'South America',
        'Chile': 'South America', 'Colombia': 'South America',
        'Australia': 'Oceania', 'New Zealand': 'Oceania',
        'South Africa': 'Africa', 'Egypt': 'Africa'
    }
    
    # Define region colors
    region_colors = {
        'North America': '#3498db',  # Blue
        'Europe': '#2ecc71',         # Green
        'Asia': '#e74c3c',           # Red
        'South America': '#f39c12',  # Orange
        'Oceania': '#9b59b6',        # Purple
        'Africa': '#1abc9c',         # Turquoise
        'Other': '#95a5a6'           # Gray
    }
    
    # Create colors list based on regions
    colors = [region_colors.get(country_to_region.get(country, 'Other'), '#95a5a6') 
             for country in plot_countries]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(plot_countries, ratio_values, color=colors)
    
    # Add a vertical line at ratio=1 (equal representation)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotations
    for i, bar in enumerate(bars):
        value = bar.get_width()
        if value > 1:
            ax.text(value + 0.1, i, f"Overrepresented ({value:.1f}x)", va='center')
        elif value < 1:
            ax.text(value + 0.1, i, f"Underrepresented ({value:.1f}x)", va='center')
        else:
            ax.text(value + 0.1, i, "Equal representation", va='center')
    
    # Add region legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=region) 
                      for region, color in region_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlabel('Representation Ratio (% of Plenary Talks / % of Parallel Talks)')
    ax.set_title('Representation Ratio in Plenary vs. Parallel Talks by Country', fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/representation_ratio.pdf')
    plt.close()

def extract_keywords_from_talk(talk):
    """Extract keywords from a talk using multiple potential formats"""
    # Define the same stopwords as in fetch_and_analyze_conferences.py
    stopwords = set(['and', 'the', 'in', 'of', 'for', 'on', 'with', 'at', 'from', 'by', 
                    'to', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'as', 'if',
                    'then', 'else', 'when', 'up', 'down', 'conference', 'study', 'analysis',
                    'measurement', 'results', 'data', 'using', 'via', 'new', 'recent',
                    'quark', 'matter', 'qm', 'physics', 'collision', 'collisions', 'ion', 'ions',
                    'heavy', 'experiment', 'experimental', 'theory', 'theoretical', 'model', 'models'])
    
    # Additional filter function matching fetch_and_analyze_conferences.py
    def is_valid_keyword(word):
        return (word.lower() not in stopwords 
                and len(word) > 2  # Skip very short words
                and not word.isdigit())  # Skip pure numbers
    
    # Try various possible keyword fields
    if 'Keywords' in talk and talk['Keywords']:
        # Filter keywords from the Keywords field
        if isinstance(talk['Keywords'], list):
            return [k for k in talk['Keywords'] if is_valid_keyword(k)]
        elif isinstance(talk['Keywords'], str):
            # Split string keywords and filter
            keywords = [k.strip() for k in talk['Keywords'].split(',')]
            return [k for k in keywords if is_valid_keyword(k)]
        return talk['Keywords']
    elif 'keywords' in talk and talk['keywords']:
        # Filter keywords from the keywords field
        if isinstance(talk['keywords'], list):
            return [k for k in talk['keywords'] if is_valid_keyword(k)]
        elif isinstance(talk['keywords'], str):
            # Split string keywords and filter
            keywords = [k.strip() for k in talk['keywords'].split(',')]
            return [k for k in keywords if is_valid_keyword(k)]
        return talk['keywords']
    elif 'tags' in talk and talk['tags']:
        # Filter tags
        if isinstance(talk['tags'], list):
            return [k for k in talk['tags'] if is_valid_keyword(k)]
        return talk['tags']
    elif 'Topic' in talk and talk['Topic']:
        # Filter topic
        if is_valid_keyword(talk['Topic']):
            return [talk['Topic']]
        return []
    
    # If no explicit keywords, try to extract from title
    if 'Title' in talk and talk['Title']:
        title = talk['Title']
        
        # Simple tokenization without NLTK
        # Remove punctuation and convert to lowercase
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        
        # Simple tokenization by splitting on whitespace
        tokens = title.split()
        
        # Filter out stopwords, short words, and pure digits
        keywords = [word for word in tokens if is_valid_keyword(word)]
        
        # Also try to extract multi-word phrases that might be keywords
        # For bigrams (two-word phrases)
        bigrams = []
        for i in range(len(tokens) - 1):
            # At least one part should not be a stopword
            if not (tokens[i].lower() in stopwords and tokens[i+1].lower() in stopwords):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                # Both parts should be valid independently
                if is_valid_keyword(tokens[i]) or is_valid_keyword(tokens[i+1]):
                    bigrams.append(bigram)
        
        # Combine individual keywords and bigrams
        potential_keywords = keywords + bigrams
        
        # Return the extracted keywords if we found any
        if potential_keywords:
            return potential_keywords
    
    # If still no keywords, try abstract as last resort
    elif 'Abstract' in talk and talk['Abstract']:
        abstract = talk['Abstract']
        if isinstance(abstract, str):
            # Try to extract a keywords section first
            keyword_section = re.search(r'keywords?:?\s*(.*?)(?:\.|$)', abstract.lower())
            if keyword_section:
                # Split by common separators
                keywords = [k.strip() for k in re.split(r'[,;]', keyword_section.group(1))]
                # Filter using same criteria
                return [k for k in keywords if k and is_valid_keyword(k)]
    
    # No keywords found
    return []

def create_keywords_plot(conference_data):
    """Create visualization of keywords across conferences based on fetch_and_analyze_conferences.py"""
    print("Creating keywords visualization...")
    
    # Common stopwords to remove
    stopwords = set(['and', 'the', 'in', 'of', 'for', 'on', 'with', 'at', 'from', 'by', 
                    'to', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'as', 'if',
                    'then', 'else', 'when', 'up', 'down', 'conference', 'study', 'analysis',
                    'measurement', 'results', 'data', 'using', 'via', 'new', 'recent',
                    'quark', 'matter', 'qm', 'physics', 'collision', 'collisions', 'ion', 'ions',
                    'heavy', 'experiment', 'experimental', 'theory', 'theoretical', 'model', 'models'])
    
    # Define physics concepts and facilities to track
    physics_concepts = {
        'flow': ['flow', 'flows', 'flowing'],
        'QGP': ['qgp', 'quark gluon plasma', 'quark-gluon'],
        'jet': ['jet', 'jets', 'dijets', 'b-jet'],
        'HF': ['heavy flavor', 'heavy flavour', 'charm', 'beauty', 'bottom', 
              'j/psi', 'jpsi', 'j/ψ', 'upsilon', 'ϒ', 'd meson', 'b meson', 
              'c-quark', 'b-quark', 'charmonium', 'bottomonium', 'open charm', 
              'open beauty', 'quarkonium', 'hf', 'charmed', 'b quark', 'c quark'],
        'photon': ['photon', 'photons', 'photonic', 'gamma'],
        'hydro': ['hydro', 'hydrodynamic', 'hydrodynamics'],
        'CME': ['cme', 'chiral magnetic', 'magnetic effect'],
        'small system': ['small system', 'small systems', 'pp collision', 'pA collision', 'p-pb', 'p+pb', 'p+a', 'p-a', 
                        'small collision', 'peripheral', 'pp collisions', 'p-p'],
        'UPC': ['upc', 'ultra-peripheral', 'ultra peripheral', 'ultraperipheral', 'photoproduction', 'coherent', 'photo-nuclear',
               'photo nuclear', 'photonuclear'],
        'soft & hard': ['soft and hard', 'hard and soft', 'soft vs hard', 'hard vs soft', 'soft versus hard', 
                       'hard versus soft', 'soft/hard', 'hard/soft']
    }
    
    facilities = {
        'RHIC': ['rhic', 'brookhaven', 'bnl'],
        'LHC': ['lhc', 'large hadron'],
        'CERN': ['cern'],
        'FAIR': ['fair', 'gsi', 'darmstadt'],
        'NICA': ['nica', 'dubna']
    }
    
    # Custom counter for "soft & hard" to catch both words in same title
    def count_concepts_with_special_cases(titles, concept_dict):
        counts = Counter()
        
        for title in titles:
            title_lower = title.lower()
            
            # Regular concept counting
            for concept, keywords in concept_dict.items():
                if concept != 'soft & hard':  # Handle regular concepts normally
                    for keyword in keywords:
                        if keyword in title_lower:
                            counts[concept] += 1
                            break  # Count each concept only once per title
            
            # Special case for "soft & hard" - both words must appear in the title
            if ('soft' in title_lower and 'hard' in title_lower):
                counts['soft & hard'] += 1
        
        return counts
    
    # Extract all years
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Prepare containers for data
    all_titles_by_year = {}
    concept_trends = {concept: [0] * len(years) for concept in physics_concepts}
    facility_trends = {facility: [0] * len(years) for facility in facilities}
    
    # Extract and analyze titles by year
    for i, year in enumerate(years):
        if year in conference_data:
            all_talks = []
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type in conference_data[year]:
                    all_talks.extend(conference_data[year][talk_type])
            
            titles = [talk.get('Title', '') for talk in all_talks if talk.get('Title')]
            all_titles_by_year[year] = titles
            
            # Only count if we have titles for this year
            if titles:
                # Count physics concepts with special case handling
                concept_counts = count_concepts_with_special_cases(titles, physics_concepts)
                for concept, count in concept_counts.items():
                    concept_trends[concept][i] = count / len(titles) * 100
                    
                # Count facility mentions (using standard function)
                facility_counts = count_concepts(titles, facilities)
                for facility, count in facility_counts.items():
                    facility_trends[facility][i] = count / len(titles) * 100
    
    # Create a figure with two subplots (only the top row now)
    try:
        # Create main visualization with only the top two panels 
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2)
        
        # Physics concepts panel
        ax_concepts = plt.subplot(gs[0, 0])
        for concept, percentages in concept_trends.items():
            ax_concepts.plot(years, percentages, marker='o', linewidth=2, label=concept)
        ax_concepts.set_title('Evolution of Physics Concepts', fontsize=14)
        ax_concepts.set_xlabel('Conference Year', fontsize=12)
        ax_concepts.set_ylabel('Percentage of Presentations (%)', fontsize=12)
        ax_concepts.grid(True, linestyle='--', alpha=0.7)
        ax_concepts.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Facilities panel
        ax_facilities = plt.subplot(gs[0, 1])
        for facility, percentages in facility_trends.items():
            ax_facilities.plot(years, percentages, marker='s', linewidth=2, label=facility)
        ax_facilities.set_title('Experimental Facilities', fontsize=14)
        ax_facilities.set_xlabel('Conference Year', fontsize=12)
        ax_facilities.set_ylabel('Percentage of Presentations (%)', fontsize=12)
        ax_facilities.grid(True, linestyle='--', alpha=0.7)
        ax_facilities.legend()
        
        plt.tight_layout()
        plt.savefig('figures/keywords_analysis.pdf', bbox_inches='tight')
        plt.savefig('figures/QM_keyword_analysis.pdf', bbox_inches='tight')  # Match existing filename
        plt.savefig('figures/keyword_trends.pdf', bbox_inches='tight')
        plt.savefig('figures/keyword_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Still process the keyword data for QA plots
        all_keywords = Counter()
        keyword_by_year = {}
        
        for year in years:
            year_keywords = Counter()
            
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type in conference_data[year]:
                    for talk in conference_data[year][talk_type]:
                        # Use the improved keyword extraction function
                        keywords = extract_keywords_from_talk(talk)
                        if keywords:
                            # Make sure keywords is a list
                            if isinstance(keywords, str):
                                keywords = [keywords]
                            
                            # Update counters
                            all_keywords.update(keywords)
                            year_keywords.update(keywords)
            
            keyword_by_year[year] = year_keywords
        
        print(f"Found {len(all_keywords)} unique keywords across all years")
        
        # Save keyword analysis as separate QA plots
        if all_keywords:
            print(f"Top 10 keywords (to be excluded): {all_keywords.most_common(10)}")
            
            # Get the top 10 keywords to exclude
            top_10_keywords = set(kw for kw, _ in all_keywords.most_common(10))
            
            # Filter out top 10 keywords
            filtered_keywords = Counter({k: v for k, v in all_keywords.items() if k not in top_10_keywords})
            
            print(f"After removing top 10, using {len(filtered_keywords)} keywords")
            print(f"New top keywords: {filtered_keywords.most_common(10)}")
            
            # Create QA plots for keyword analysis (saved separately)
            fig_qa = plt.figure(figsize=(20, 10))
            gs_qa = gridspec.GridSpec(1, 2)
            
            # Bar chart of top keywords
            ax_keywords = plt.subplot(gs_qa[0, 0])
            top_keywords = [kw for kw, _ in filtered_keywords.most_common(15)]
            counts = [filtered_keywords[kw] for kw in top_keywords]
            
            y_pos = range(len(top_keywords))
            ax_keywords.barh(y_pos, counts)
            ax_keywords.set_yticks(y_pos)
            ax_keywords.set_yticklabels([kw if len(kw) < 30 else kw[:27]+'...' for kw in top_keywords])
            ax_keywords.set_xlabel('Frequency')
            ax_keywords.set_title('Top Keywords (excluding 10 most common)')
            
            # Keyword trends
            ax_keyword_trends = plt.subplot(gs_qa[0, 1])
            top_trend_keywords = top_keywords[:6]
            
            for keyword in top_trend_keywords:
                trend = []
                for year in years:
                    trend.append(keyword_by_year.get(year, Counter()).get(keyword, 0))
                ax_keyword_trends.plot(years, trend, 'o-', linewidth=2, label=keyword)
            
            ax_keyword_trends.set_title('Keyword Trends Over Time')
            ax_keyword_trends.set_xlabel('Conference Year')
            ax_keyword_trends.set_ylabel('Frequency')
            ax_keyword_trends.legend()
            ax_keyword_trends.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('figures/keyword_QA_plots.pdf', bbox_inches='tight')
            plt.savefig('figures/keyword_QA_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Filter the keyword_by_year data 
        filtered_keyword_by_year = {}
        for year in keyword_by_year:
            filtered_keyword_by_year[year] = Counter({k: v for k, v in keyword_by_year[year].items() 
                                                      if k not in top_10_keywords})
        
        return filtered_keywords, filtered_keyword_by_year
    
    except Exception as e:
        print(f"Error creating keywords visualization: {e}")
        traceback.print_exc()
        
        # Create simple fallback visualizations in case of error
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating keyword visualization:\n{str(e)}", 
                ha='center', va='center', fontsize=12)
        plt.savefig('figures/keywords_analysis.pdf')
        plt.savefig('figures/QM_keyword_analysis.pdf')
        plt.savefig('figures/keyword_trends.pdf')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating keyword visualization:\n{str(e)}", 
                ha='center', va='center', fontsize=12)
        plt.savefig('figures/keyword_trends.png')
        plt.close()
        
        return None, None

def count_concepts(titles, concept_dict):
    """Count occurrences of concepts in talk titles"""
    counts = Counter()
    
    for title in titles:
        title_lower = title.lower()
        for concept, keywords in concept_dict.items():
            for keyword in keywords:
                if keyword in title_lower:
                    counts[concept] += 1
                    break  # Count each concept only once per title
    
    return counts

def create_plenary_country_plot(plenary_country):
    """Create a visualization of countries in plenary talks"""
    print("Creating plenary country distribution visualization...")
    
    # Check if we have data
    if not plenary_country:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No plenary country data available", 
                ha='center', va='center', fontsize=16)
        plt.savefig('figures/plenary_country_distribution.pdf')
        plt.close()
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get top countries
    top_countries = [country for country, _ in plenary_country.most_common(15) 
                     if country != 'Unknown']
    counts = [plenary_country[country] for country in top_countries]
    
    # Create bar chart
    y_pos = range(len(top_countries))
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, top_countries)
    plt.xlabel('Number of Plenary Talks')
    plt.title('Countries Represented in Plenary Talks')
    
    plt.tight_layout()
    plt.savefig('figures/plenary_country_distribution.pdf')
    plt.savefig('figures/plenary_talks_by_country.pdf')  # Match existing filename
    plt.close()

def create_representation_ratio_plot(plenary_country, parallel_country):
    """Create visualization showing representation ratio between plenary and parallel talks"""
    print("Creating representation ratio visualization...")
    
    # Check if we have enough data
    if not plenary_country or not parallel_country:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Insufficient data for representation ratio", 
                ha='center', va='center', fontsize=16)
        plt.savefig('figures/representation_ratio_by_year.pdf')
        plt.close()
        return
    
    # Get top countries across both categories
    all_countries = Counter()
    all_countries.update(plenary_country)
    all_countries.update(parallel_country)
    
    top_countries = [country for country, _ in all_countries.most_common(10) 
                     if country != 'Unknown']
    
    # Calculate ratios
    ratios = []
    names = []
    
    plenary_total = sum(plenary_country.values())
    parallel_total = sum(parallel_country.values())
    
    for country in top_countries:
        plenary_pct = 100 * plenary_country.get(country, 0) / plenary_total if plenary_total > 0 else 0
        parallel_pct = 100 * parallel_country.get(country, 0) / parallel_total if parallel_total > 0 else 0
        
        if parallel_pct > 0:
            ratio = plenary_pct / parallel_pct
            ratios.append(ratio)
            names.append(country)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Sort by ratio
    sorted_indices = np.argsort(ratios)
    sorted_ratios = [ratios[i] for i in sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    
    # Colors indicating over/under representation
    colors = ['red' if r < 1 else 'green' for r in sorted_ratios]
    
    plt.barh(range(len(sorted_ratios)), sorted_ratios, color=colors)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Plenary/Parallel Representation Ratio')
    plt.title('Representation Ratio in Plenary vs Parallel Talks')
    
    # Add annotations
    for i, v in enumerate(sorted_ratios):
        plt.text(v + 0.05, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig('figures/representation_ratio_by_year.pdf')
    plt.close()

def create_regional_diversity_plot(country_counts, conference_data):
    """Create plot showing regional diversity"""
    print("Creating regional diversity visualization...")
    
    # Define regions
    regions = {
        'North America': ['United States', 'USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Netherlands', 
                  'Poland', 'Spain', 'Sweden', 'Finland', 'Norway', 'Denmark', 
                  'Czech Republic', 'Hungary', 'Austria', 'Belgium', 'Portugal',
                  'Greece', 'Romania', 'Serbia', 'Croatia', 'Slovakia', 'Slovenia',
                  'Bulgaria', 'Ireland', 'Ukraine', 'Russia'],
        'Asia': ['China', 'Japan', 'India', 'South Korea', 'Taiwan', 'Singapore',
                'Israel', 'Turkey', 'Iran', 'Pakistan', 'Malaysia', 'Thailand'],
        'Other': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Mexico',
                 'South Africa', 'Australia', 'New Zealand']
    }
    
    # Calculate regional totals
    region_counts = Counter()
    
    for country, count in country_counts.items():
        if country == 'Unknown':
            continue
            
        found = False
        for region, countries in regions.items():
            if country in countries:
                region_counts[region] += count
                found = True
                break
                
        if not found:
            region_counts['Other'] += count
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    
    # Sort regions by count
    sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_regions) if sorted_regions else ([], [])
    
    # Colors
    colors = plt.cm.tab10.colors[:len(labels)]
    
    # Create pie chart with percentages
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, shadow=False)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Regional Distribution of Contributions')
    
    plt.tight_layout()
    plt.savefig('figures/regional_diversity.pdf')
    plt.close()
    
    # Also create regional diversity by year plot if we have multiple years
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    if len(years) > 1:
        create_regional_diversity_by_year(years, conference_data, regions)

def create_regional_diversity_by_year(years, conference_data, regions):
    """Create plot showing regional diversity by year"""
    print("Creating regional diversity by year visualization...")
    
    # Calculate regional distribution by year
    region_by_year = {region: [] for region in regions.keys()}
    region_by_year['Other'] = []  # Ensure 'Other' category exists
    
    for year in years:
        # Collect country counts for this year
        year_country_counts = Counter()
        
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                for talk in conference_data[year][talk_type]:
                    country = talk.get('Country', 'Unknown')
                    if country != 'Unknown':
                        year_country_counts[country] += 1
        
        # Calculate regional distribution
        year_region_counts = Counter()
        
        for country, count in year_country_counts.items():
            found = False
            for region, countries in regions.items():
                if country in countries:
                    year_region_counts[region] += count
                    found = True
                    break
            if not found:
                year_region_counts['Other'] += count
        
        # Calculate percentages
        total = sum(year_region_counts.values())
        
        for region in region_by_year:
            if total > 0:
                region_by_year[region].append(year_region_counts.get(region, 0) / total * 100)
            else:
                region_by_year[region].append(0)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot regional trends
    for region, percentages in region_by_year.items():
        if sum(percentages) > 0:  # Only plot regions with data
            plt.plot(years, percentages, marker='o', linewidth=2, label=region)
    
    plt.xlabel('Conference Year')
    plt.ylabel('Percentage of Contributions')
    plt.title('Regional Representation Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/regional_diversity_by_year.pdf')
    plt.close()

def plot_talks_by_institute(talks, title, filename):
    """
    Create a bar chart showing the top institutes by number of talks.
    
    Parameters:
    - talks: List of talk data
    - title: Title for the plot
    - filename: Filename to save the plot
    """
    print(f"Creating institute visualization for {title}...")
    
    # Count talks by institute
    institute_counts = Counter()
    
    # Tracking for debugging
    missing_institute_count = 0
    
    # Check for different field names that might contain institute information
    for talk in talks:
        institute = None
        
        # Try different possible field names for institute information
        for field in ['Institute', 'Affiliation', 'institution', 'affiliation']:
            if field in talk and talk[field] and talk[field] != 'Unknown':
                institute = talk[field]
                break
        
        # Clean up institute name if needed
        if institute:
            # Normalize the institute name
            institute = normalize_institute_name(institute)
            
            # Add to counts
            institute_counts[institute] += 1
        else:
            missing_institute_count += 1
    
    print(f"  Found {len(institute_counts)} unique institutes for {sum(institute_counts.values())} talks")
    print(f"  {missing_institute_count} talks were missing institute information")
    
    # Check if we found any institutes
    if not institute_counts:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"No institute data found for {title}", 
                ha='center', va='center', fontsize=16)
        plt.savefig(f'figures/{filename}')
        plt.close()
        return
    
    # Get top institutes (limit to 25 for readability)
    top_institutes = institute_counts.most_common(25)
    
    # Debug print to see what we found
    print(f"  Top 5 institutes: {top_institutes[:5]}")
    
    # Prepare data for horizontal bar chart
    institutes = []
    counts = []
    
    for institute, count in top_institutes:
        if len(institute) > 40:  # Truncate very long names
            institute = institute[:37] + '...'
        institutes.append(institute)
        counts.append(count)
    
    # Create horizontal bar chart
    plt.figure(figsize=(14, 10))
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(institutes)))
    
    # Plot bars
    bars = plt.barh(range(len(institutes)), counts, color=colors, alpha=0.8)
    
    # Add count labels to the right of each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Number of Talks')
    plt.title(f'{title}: Top Institutes')
    
    # Set y-ticks (institute names)
    plt.yticks(range(len(institutes)), institutes)
    
    # Add grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}')
    plt.close()

def normalize_institute_name(name):
    """
    Normalize institute names to group similar entries.
    
    Parameters:
    - name: Raw institute name
    
    Returns:
    - Normalized institute name
    """
    # Convert to string if not already
    if not isinstance(name, str):
        return "Unknown"
    
    # Convert to lowercase for case-insensitive comparison
    normalized = name.lower()
    
    # Remove common appendages and punctuation
    for pattern in [', inc.', ', ltd.', ', llc', ', gmbh', ', corp', ', corporation', 
                    ' incorporated', ' limited', ' corporation', ' university']:
        normalized = normalized.replace(pattern.lower(), '')
    
    # Normalize common abbreviations
    abbr_map = {
        'univ': 'university',
        'inst': 'institute',
        'lab': 'laboratory',
        'natl': 'national',
        'dept': 'department',
        'coll': 'college',
        'tech': 'technology',
        'sci': 'science',
        'phys': 'physics',
        'sch': 'school',
        'assoc': 'association',
        'res': 'research',
        'acad': 'academy',
        'center': 'centre',  # Standardize US/UK spelling
        'center': 'centre'
    }
    
    # Apply abbreviation normalization
    for abbr, full in abbr_map.items():
        # Look for the abbreviation as a standalone word
        normalized = re.sub(r'\b' + abbr + r'\b', full, normalized)
        # Also replace when followed by a period
        normalized = normalized.replace(abbr + '.', full)
    
    # Special cases for major institutions
    institution_map = {
        'mit': 'Massachusetts Institute of Technology',
        'lbl': 'Lawrence Berkeley National Laboratory',
        'cern': 'CERN',
        'ornl': 'Oak Ridge National Laboratory',
        'bnl': 'Brookhaven National Laboratory',
        'jlab': 'Jefferson Laboratory',
        'fnal': 'Fermi National Accelerator Laboratory',
        'lanl': 'Los Alamos National Laboratory',
        'pnnl': 'Pacific Northwest National Laboratory',
        'slac': 'SLAC National Accelerator Laboratory'
    }
    
    # Check for major institution names
    for abbr, full in institution_map.items():
        if abbr == normalized or abbr + ' ' in normalized or ' ' + abbr in normalized:
            return full
    
    # Capitalize properly and return
    words = normalized.split()
    if words:
        return ' '.join(word.capitalize() for word in words)
    else:
        return "Unknown"

def analyze_institute_diversity(conference_data):
    """Analyze institute diversity across conferences"""
    print("Analyzing institute diversity...")
    
    # Extract all years
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Create aggregate visualization for all talk types combined
    try:
        all_talks = []
        for year in years:
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type in conference_data[year]:
                    all_talks.extend(conference_data[year][talk_type])
        
        if all_talks:
            plot_talks_by_institute(
                all_talks,
                "All Talks",
                "all_institutes.pdf"
            )
    except Exception as e:
        print(f"Error creating aggregate institute visualization: {e}")
        traceback.print_exc()
    
    # Create visualizations for individual talk types (without yearly breakdowns)
    for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
        try:
            # Collect all talks of this type across years
            all_talks = []
            for year in years:
                if talk_type in conference_data[year]:
                    all_talks.extend(conference_data[year][talk_type])
            
            # Create visualization for all years combined
            if all_talks:
                plot_talks_by_institute(
                    all_talks,
                    f"All Years {talk_type.replace('_', ' ').title()}",
                    f"all_years_{talk_type}_by_institute.pdf"
                )
            
        except Exception as e:
            print(f"Error creating {talk_type} institute visualization: {e}")
            traceback.print_exc()

def debug_data_structure(conference_data):
    """Print out a sample of the data structure to help debug issues"""
    print("\n===== DEBUGGING DATA STRUCTURE =====")
    
    # Check overall structure
    print(f"Years in data: {sorted(conference_data.keys())}")
    
    # Check a sample year
    sample_year = next(iter(conference_data.keys()))
    print(f"\nSample year: {sample_year}")
    print(f"Keys in year data: {list(conference_data[sample_year].keys())}")
    
    # Check talk types
    for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
        if talk_type in conference_data[sample_year]:
            print(f"\nFound {len(conference_data[sample_year][talk_type])} {talk_type}")
            
            # Look at the first talk
            if conference_data[sample_year][talk_type]:
                sample_talk = conference_data[sample_year][talk_type][0]
                print(f"Sample {talk_type} keys: {list(sample_talk.keys())}")
                
                # Check specifically for keywords
                if 'Keywords' in sample_talk:
                    print(f"Keywords found: {sample_talk['Keywords']}")
                elif 'keywords' in sample_talk:
                    print(f"keywords (lowercase) found: {sample_talk['keywords']}")
                elif 'tags' in sample_talk:
                    print(f"tags found: {sample_talk['tags']}")
                else:
                    # Check if there's anything that might contain keywords
                    for key, value in sample_talk.items():
                        if isinstance(value, (list, str)) and ('keyword' in key.lower() or 'tag' in key.lower()):
                            print(f"Possible keyword field: {key}: {value}")
    
    print("===== END DEBUGGING =====\n")

def analyze_country_distribution(conference_data):
    """Analyze country distribution across conferences"""
    print("Analyzing country distribution...")
    
    # Get all countries across all years
    country_counts = Counter()
    
    for year, data in conference_data.items():
        year_country_counts = Counter()
        
        # Count countries for each talk type
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    country = talk.get('Country', 'Unknown')
                    if country != 'Unknown':
                        year_country_counts[country] += 1
        
        # Update overall counts
        country_counts.update(year_country_counts)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    top_countries = [country for country, count in country_counts.most_common(15)]
    top_counts = [country_counts[country] for country in top_countries]
    
    # Create horizontal bar chart
    y_pos = range(len(top_countries))
    plt.barh(y_pos, top_counts, align='center')
    plt.yticks(y_pos, top_countries)
    plt.xlabel('Number of Talks')
    plt.title('Top Countries by Number of Talks')
    
    plt.tight_layout()
    plt.savefig('figures/country_distribution.pdf')
    plt.savefig('figures/top_countrys.pdf')  # Match existing filename
    plt.close()
    
    # Create another plot for trends over time
    plt.figure(figsize=(14, 8))
    
    # Get sorted years
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Get top countries to track
    top_countries_to_track = [country for country, _ in country_counts.most_common(8)]
    
    # Track countries over time
    for country in top_countries_to_track:
        country_by_year = []
        for year in years:
            year_count = 0
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type in conference_data[year]:
                    year_count += sum(1 for talk in conference_data[year][talk_type] 
                                    if talk.get('Country', 'Unknown') == country)
            country_by_year.append(year_count)
        
        plt.plot(years, country_by_year, 'o-', linewidth=2, label=country)
    
    plt.title('Country Trends Over Time')
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Talks')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/country_trends.pdf')
    plt.savefig('figures/country_trends_over_time.pdf')  # Match existing filename
    plt.close()
    
    return country_counts

def analyze_plenary_vs_parallel(conference_data):
    """Analyze plenary vs parallel talks by country"""
    print("Analyzing plenary vs parallel talks...")
    
    # Count countries for plenary and parallel talks
    plenary_country = Counter()
    parallel_country = Counter()
    
    for year, data in conference_data.items():
        # Count plenary talks
        if 'plenary_talks' in data:
            for talk in data['plenary_talks']:
                country = talk.get('Country', 'Unknown')
                if country != 'Unknown':
                    plenary_country[country] += 1
        
        # Count parallel talks
        if 'parallel_talks' in data:
            for talk in data['parallel_talks']:
                country = talk.get('Country', 'Unknown')
                if country != 'Unknown':
                    parallel_country[country] += 1
    
    return plenary_country, parallel_country

def create_plenary_country_plot(plenary_country):
    """Create plot showing distribution of plenary talks by country"""
    print("Creating plenary country distribution plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Get top countries for plenary talks
    top_plenary = [country for country, count in plenary_country.most_common(10)]
    counts = [plenary_country[country] for country in top_plenary]
    
    # Create bar chart
    y_pos = range(len(top_plenary))
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, top_plenary)
    plt.xlabel('Number of Plenary Talks')
    plt.title('Top Countries for Plenary Talks')
    
    plt.tight_layout()
    plt.savefig('figures/plenary_country_distribution.pdf')
    plt.savefig('figures/plenary_talks_by_country.pdf')  # Match existing filename
    plt.close()

def create_representation_ratio_plot(plenary_country, parallel_country):
    """Create plot showing representation ratio by country"""
    print("Creating representation ratio plot...")
    
    # Calculate representation ratio
    # (% of plenary talks) / (% of parallel talks)
    total_plenary = sum(plenary_country.values())
    total_parallel = sum(parallel_country.values())
    
    ratio_by_country = {}
    
    # Get countries that have at least 5 parallel talks
    significant_countries = [country for country, count in parallel_country.items() 
                           if count >= 5]
    
    for country in significant_countries:
        plenary_percent = (plenary_country.get(country, 0) / total_plenary) * 100
        parallel_percent = (parallel_country.get(country, 0) / total_parallel) * 100
        
        if parallel_percent > 0:  # Avoid division by zero
            ratio = plenary_percent / parallel_percent
            ratio_by_country[country] = ratio
    
    # Sort by ratio
    sorted_countries = sorted(ratio_by_country.items(), key=lambda x: x[1], reverse=True)
    countries, ratios = zip(*sorted_countries)
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    
    y_pos = range(len(countries))
    plt.bar(y_pos, ratios)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)  # Line at ratio = 1
    plt.xticks(y_pos, countries, rotation=45, ha='right')
    plt.ylabel('Representation Ratio')
    plt.title('Country Representation Ratio (Plenary/Parallel)')
    plt.ylim(0, max(3.0, max(ratios) * 1.1))  # Cap at 3x or the max value
    
    plt.tight_layout()
    plt.savefig('figures/representation_ratio.pdf')
    plt.savefig('figures/representation_ratio_by_year.pdf')  # Match existing filename
    plt.close()

def load_processed_data():
    """Load the processed conference data from JSON file"""
    try:
        with open('data/processed_conference_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def display_conference_summary(conference_data):
    """Display summary of conference data"""
    for year, data in sorted(conference_data.items()):
        print(f"\nYear: {year}")
        for talk_type in data:
            if isinstance(data[talk_type], list):
                print(f"  {talk_type}: {len(data[talk_type])} talks")

def fix_unknown_institute_country_data(conference_data):
    """Fix unknown institute and country data"""
    # Placeholder implementation - you'd need to implement this properly
    return conference_data

def fix_common_affiliation_problems(conference_data):
    """Fix common affiliation problems"""
    # Placeholder implementation - you'd need to implement this properly
    return conference_data

def add_manual_country_fixes(conference_data):
    """Add manual country fixes"""
    # Placeholder implementation - you'd need to implement this properly
    return conference_data

def fix_unknown_institutes(conference_data):
    """Fix unknown institutes"""
    # Placeholder implementation - you'd need to implement this properly
    return conference_data

def filter_relevant_talk_types(conference_data):
    """
    Filter conference data to only include plenary, parallel, and poster talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Updated conference_data with only relevant talk types
    """
    print("\nFiltering data to include only plenary, parallel, and poster talks...")
    
    for year, data in conference_data.items():
        # Create new all_talks list with only the relevant talk types
        all_talks = []
        
        # Add all plenary talks
        if 'plenary_talks' in data:
            all_talks.extend(data['plenary_talks'])
        
        # Add all parallel talks
        if 'parallel_talks' in data:
            all_talks.extend(data['parallel_talks'])
        
        # Add all poster talks
        if 'poster_talks' in data:
            all_talks.extend(data['poster_talks'])
        
        # Update the all_talks entry
        data['all_talks'] = all_talks
        
        # Remove other talk types if they exist
        if 'flash_talks' in data:
            print(f"  Removed {len(data['flash_talks'])} flash talks from QM{year}")
            del data['flash_talks']
        
        if 'other_talks' in data:
            print(f"  Removed {len(data['other_talks'])} other talks from QM{year}")
            del data['other_talks']
    
    return conference_data

def analyze_country_diversity(filtered_data):
    """Analyze country diversity metrics"""
    years = sorted([year for year in filtered_data.keys() if year != '2025'])
    
    # Track metrics by year
    unique_countries = {}
    hhi_by_year = {}  # Herfindahl-Hirschman Index - measure of concentration
    
    for year in years:
        # Count countries
        country_counts = Counter()
        
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in filtered_data[year]:
                for talk in filtered_data[year][talk_type]:
                    country = talk.get('Country', 'Unknown')
                    if country != 'Unknown':
                        country_counts[country] += 1
        
        # Calculate metrics
        unique_countries[year] = len(country_counts)
        
        # Calculate HHI
        total_talks = sum(country_counts.values())
        if total_talks > 0:
            hhi = sum((count/total_talks)**2 for count in country_counts.values())
            hhi_by_year[year] = hhi * 10000  # Scale for better visualization
        else:
            hhi_by_year[year] = 0
    
    return unique_countries, hhi_by_year

def create_diversity_metrics_plot(years, unique_countries, hhi_by_year):
    """Create plot showing diversity metrics over time"""
    print("Creating diversity metrics plot...")
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot unique countries
    color = 'tab:blue'
    ax1.set_xlabel('Conference Year')
    ax1.set_ylabel('Number of Unique Countries', color=color)
    unique_values = [unique_countries.get(year, 0) for year in years]
    ax1.plot(years, unique_values, 'o-', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot HHI on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('HHI (lower = more diverse)', color=color)
    hhi_values = [hhi_by_year.get(year, 0) for year in years]
    ax2.plot(years, hhi_values, 's--', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Country Diversity Metrics Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/diversity_metrics.pdf')
    plt.savefig('figures/data_quality.pdf')  # Match existing filename
    plt.close()

def normalize_institute_name(name):
    """Normalize institute names for consistent identification"""
    # Basic normalization
    name = name.strip().lower()
    
    # Handle common variations and abbreviations
    mappings = {
        'cern': 'CERN',
        'brookhaven': 'BNL',
        'bnl': 'BNL',
        'lawrence berkeley': 'LBNL',
        'lbnl': 'LBNL',
        'berkeley lab': 'LBNL',
        'mit': 'MIT',
        'berkeley': 'UC Berkeley',
        'los alamos': 'LANL',
        'lanl': 'LANL',
        'oak ridge': 'ORNL',
        'ornl': 'ORNL',
        'argonne': 'ANL',
        'anl': 'ANL',
        'jyväskylä': 'University of Jyväskylä',
        'jyvaskyla': 'University of Jyväskylä',
        'university of jyväskylä': 'University of Jyväskylä',
        'university of jyvaskyla': 'University of Jyväskylä',
        'unam': 'UNAM',
        'gsi': 'GSI',
        'infn': 'INFN',
        'dubna': 'JINR',
        'jinr': 'JINR'
    }
    
    # Apply mappings
    for key, value in mappings.items():
        if key in name:
            return value
    
    # Return the original name with first letters capitalized for readability
    return ' '.join(word.capitalize() for word in name.split())

def create_institute_bubble_chart(conference_data):
    """
    Create a bubble chart showing institute contributions across conference years.
    
    This visualization shows the top 30 institutes and their contributions over time,
    with bubble size proportional to number of presentations.
    """
    print("Creating institute bubble chart...")
    
    # Extract years, excluding 2025 if present
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Count institutes across all years
    all_institute_counts = Counter()
    
    # Dictionary to store institute counts by year
    institute_by_year = {}
    
    # First, count all institutes across all years
    for year in years:
        institute_counts = Counter()
        
        # Get all talks for this year
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                for talk in conference_data[year][talk_type]:
                    # Try different possible fields for institute
                    institute = None
                    for field in ['Institute', 'Affiliation', 'institution', 'affiliation']:
                        if field in talk and talk[field] and talk[field] != 'Unknown':
                            institute = normalize_institute_name(talk[field])
                            break
                    
                    if institute:
                        institute_counts[institute] += 1
        
        # Add to overall counts
        all_institute_counts.update(institute_counts)
        
        # Store this year's counts
        institute_by_year[year] = institute_counts
    
    # Get top 30 institutes by total count
    top_institutes = [inst for inst, count in all_institute_counts.most_common(30)]
    
    # Create a matrix for the bubble chart
    matrix = []
    for institute in top_institutes:
        row = []
        for year in years:
            row.append(institute_by_year[year].get(institute, 0))
        matrix.append(row)
    
    # Create the bubble chart
    plt.figure(figsize=(14, 12))
    
    # Create meshgrid for bubble positions
    x_indices = np.arange(len(years))
    y_indices = np.arange(len(top_institutes))
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Flatten the arrays for scatter plot
    x = X.flatten()
    y = Y.flatten()
    
    # Get bubble sizes from matrix
    sizes = []
    for i, row in enumerate(matrix):
        for count in row:
            sizes.append(count * 50)  # Scale size for visibility
    
    # Create scatter plot
    plt.scatter(x, y, s=sizes, alpha=0.7, edgecolors='gray', linewidth=1)
    
    # Set axes labels and ticks
    plt.xlabel('Conference Year')
    plt.ylabel('Institute')
    plt.xticks(x_indices, years)
    plt.yticks(y_indices, top_institutes)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    plt.title('Institute Contributions Across Conference Years')
    
    plt.tight_layout()
    plt.savefig('figures/institute_bubble_chart.pdf', bbox_inches='tight')
    plt.close()

def analyze_conference_data():
    """Main function to analyze conference data"""
    try:
        # Load the processed data
        conference_data = load_processed_data()
        if not conference_data:
            print("Error: Could not load processed data")
            return
            
        print("\n===== BEGINNING ANALYSIS =====")
        
        # Create output directories
        os.makedirs('figures', exist_ok=True)
        os.makedirs('data/analysis', exist_ok=True)
        
        # Display initial conference summary
        print("\nConference summary from processed data:")
        display_conference_summary(conference_data)
        
        # STEP 1: Fix inconsistencies in institute and country data
        print("\nSTEP 1: Fixing inconsistencies in institute and country data...")
        conference_data = fix_unknown_institute_country_data(conference_data)
        
        # STEP 2: Fix common affiliation problems
        print("\nSTEP 2: Fixing common affiliation problems...")
        conference_data = fix_common_affiliation_problems(conference_data)
        
        # STEP 3: Add manual country fixes
        print("\nSTEP 3: Adding manual country fixes...")
        conference_data = add_manual_country_fixes(conference_data)
        
        # STEP 4: Specifically fix unknown institutes
        print("\nSTEP 4: Specifically fixing unknown institutes...")
        conference_data = fix_unknown_institutes(conference_data)
        
        # Display final conference summary
        print("\nFinal conference summary:")
        display_conference_summary(conference_data)
        
        # STEP 5: Filter to only include relevant talk types
        print("\nSTEP 5: Filtering to include only relevant talk types...")
        filtered_data = filter_relevant_talk_types(conference_data)
        if not filtered_data:
            print("Error: Filtering failed, using original data")
            filtered_data = conference_data
        
        # Create QM talk statistics figure with the filtered data
        print("\nCreating QM talk statistics figure...")
        try:
            create_talk_statistics_figure(filtered_data)
            print("QM talk statistics figure created successfully!")
        except Exception as e:
            print(f"Error creating QM talk statistics figure: {e}")
            traceback.print_exc()
            
        # STEP 6: Generate all visualizations for the paper
        print("\nSTEP 6: Generating visualizations for the paper...")
        
        # Add gender diversity analysis and visualization
        try:
            print("Analyzing gender diversity...")
            gender_by_year, gender_by_talk_type = analyze_gender_diversity(filtered_data)
            create_gender_diversity_plot(gender_by_year, gender_by_talk_type)
            print("Gender diversity visualization created successfully!")
        except Exception as e:
            print(f"Error analyzing gender diversity: {e}")
            traceback.print_exc()
        
        # Figure: Keywords visualization 
        try:
            print("Creating keywords visualization...")
            create_keywords_plot(filtered_data)
        except Exception as e:
            print(f"Error creating keywords visualization: {e}")
            traceback.print_exc()
            
        # Figures: Country analysis
        try:
            print("Analyzing country distribution...")
            country_counts = analyze_country_distribution(filtered_data)
        except Exception as e:
            print(f"Error analyzing country distribution: {e}")
            traceback.print_exc()
            
        # Figures: Plenary vs parallel talks
        try:
            print("Analyzing plenary vs parallel talks...")
            plenary_country, parallel_country = analyze_plenary_vs_parallel(filtered_data)
            
            # Create plenary country visualization
            try:
                print("Creating plenary country visualization...")
                create_plenary_country_plot(plenary_country)
            except Exception as e:
                print(f"Error creating plenary country visualization: {e}")
                traceback.print_exc()
            
            # Create parallel country visualization
            try:
                print("Creating parallel country visualization...")
                create_parallel_country_plot(parallel_country)
            except Exception as e:
                print(f"Error creating parallel country visualization: {e}")
                traceback.print_exc()
            
            # Create representation ratio visualization
            try:
                print("Creating representation ratio visualization...")
                create_representation_ratio_plot(plenary_country, parallel_country)
            except Exception as e:
                print(f"Error creating representation ratio: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error analyzing plenary vs parallel talks: {e}")
            traceback.print_exc()
            
        # Create regional diversity visualization
        try:
            print("Creating regional diversity visualization...")
            create_regional_diversity_plot(country_counts, filtered_data)
        except Exception as e:
            print(f"Error creating regional diversity visualization: {e}")
            traceback.print_exc()
            
        # Create institute visualizations
        try:
            print("Creating institute visualizations...")
            analyze_institute_diversity(filtered_data)
        except Exception as e:
            print(f"Error creating institute visualizations: {e}")
            traceback.print_exc()
            
        # Create institute bubble chart
        try:
            print("Creating institute bubble chart...")
            create_institute_bubble_chart(filtered_data)
        except Exception as e:
            print(f"Error creating institute bubble chart: {e}")
            traceback.print_exc()
        
        print("\n===== ANALYSIS COMPLETE =====")
        print("All visualizations have been saved to the 'figures' directory")
        
        return conference_data
    
    except Exception as e:
        print(f"Error in analyze_conference_data: {e}")
        traceback.print_exc()
        return None

def create_talk_statistics_figure(conference_data):
    """
    Create a comprehensive plot of QM conference statistics.
    Based on the implementation in fetch_and_analyze_conferences.py.
    
    Parameters:
    - conference_data: Dictionary with conference data by year
    """
    print("Creating QM talk statistics figure...")
    
    # Extract all years 
    years = sorted([int(year) for year in conference_data.keys() if year != '2025' and year.isdigit()])
    x = np.array(years)
    
    # Set up the plot
    fig = plt.figure(figsize=(14, 10))
    
    # Create GridSpec for layout
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 1])
    
    # Plot 1: Talk Counts by Type
    ax1 = plt.subplot(gs[0])
    plenary_counts = []
    parallel_counts = []
    poster_counts = []
    
    for year in years:
        year_str = str(year)
        if year_str in conference_data:
            plenary_counts.append(len(conference_data[year_str].get('plenary_talks', [])))
            parallel_counts.append(len(conference_data[year_str].get('parallel_talks', [])))
            poster_counts.append(len(conference_data[year_str].get('poster_talks', [])))
    
    # Create the stacked bar chart for talk counts
    width = 0.8
    bars1 = ax1.bar(x, plenary_counts, width, label='Plenary Talks', color='#1f77b4')
    bars2 = ax1.bar(x, parallel_counts, width, label='Parallel Talks', bottom=plenary_counts, color='#ff7f0e')
    
    # Calculate cumulative heights for stacking
    bottom = np.array(plenary_counts) + np.array(parallel_counts)
    bars3 = ax1.bar(x, poster_counts, width, label='Posters', bottom=bottom, color='#2ca02c')
    
    # Add legend
    ax1.legend(loc='upper left')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel('Number of Talks')
    ax1.set_title('QM Conference Statistics by Year', fontsize=16)
    
    # Add annotations for total talk counts
    for i in range(len(x)):
        total = plenary_counts[i] + parallel_counts[i] + poster_counts[i]
        ax1.text(x[i], total + 5, f"{total}", ha='center', va='bottom')
    
    # Calculate unique countries and institutes per year
    countries_by_year = []
    institutes_by_year = []
    
    # Plot 2: Country diversity
    ax2 = plt.subplot(gs[1])
    
    for year in years:
        year_str = str(year)
        countries = set()
        institutes = set()
        
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year_str]:
                for talk in conference_data[year_str][talk_type]:
                    if 'Country' in talk and talk['Country'] != 'Unknown':
                        countries.add(talk['Country'])
                    
                    for field in ['Institute', 'Affiliation', 'institution', 'affiliation']:
                        if field in talk and talk[field] and talk[field] != 'Unknown':
                            institutes.add(normalize_institute_name(talk[field]))
                            break
        
        countries_by_year.append(len(countries))
        institutes_by_year.append(len(institutes))
    
    # Plot country diversity
    ax2.plot(x, countries_by_year, 'o-', color='#d62728', linewidth=2, label='Unique Countries')
    ax2.set_ylabel('Number of Countries', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Add secondary y-axis for institutes
    ax2_2 = ax2.twinx()
    ax2_2.plot(x, institutes_by_year, 's-', color='#9467bd', linewidth=2, label='Unique Institutes')
    ax2_2.set_ylabel('Number of Institutes', color='#9467bd')
    ax2_2.tick_params(axis='y', labelcolor='#9467bd')
    
    # Combine legends for both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_title('Geographic Diversity in QM Conferences', fontsize=16)
    
    # Plot 3: Talk type distribution over time
    ax3 = plt.subplot(gs[2])
    
    # Calculate percentages
    total_talks = np.array(plenary_counts) + np.array(parallel_counts) + np.array(poster_counts)
    plenary_pct = np.array(plenary_counts) / total_talks * 100
    parallel_pct = np.array(parallel_counts) / total_talks * 100
    poster_pct = np.array(poster_counts) / total_talks * 100
    
    ax3.plot(x, plenary_pct, 'o-', color='#1f77b4', linewidth=2, label='Plenary %')
    ax3.plot(x, parallel_pct, 's-', color='#ff7f0e', linewidth=2, label='Parallel %')
    ax3.plot(x, poster_pct, '^-', color='#2ca02c', linewidth=2, label='Poster %')
    
    ax3.set_xlabel('Conference Year')
    ax3.set_ylabel('Percentage of Talks')
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.set_title('Talk Type Distribution Over Time', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='center left')
    
    plt.tight_layout()
    
    # Save the figure with explicit paths and names
    print("Saving QM_talk_statistics.pdf...")
    plt.savefig('figures/QM_talk_statistics.pdf', bbox_inches='tight')
    plt.savefig('figures/QM_talk_statistics.png', dpi=300, bbox_inches='tight')
    
    # Make sure the figure is closed properly
    plt.close(fig)
    print("QM talk statistics figure saved successfully!")

def create_parallel_country_plot(parallel_country):
    """Create plot showing distribution of parallel talks by country"""
    print("Creating parallel country distribution plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Get top countries for parallel talks
    top_parallel = [country for country, count in parallel_country.most_common(10)]
    counts = [parallel_country[country] for country in top_parallel]
    
    # Create bar chart
    y_pos = range(len(top_parallel))
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, top_parallel)
    plt.xlabel('Number of Parallel Talks')
    plt.title('Top Countries for Parallel Talks')
    
    plt.tight_layout()
    plt.savefig('figures/parallel_talks_by_country.pdf')
    plt.close()

def estimate_gender_from_name(first_name):
    """
    Estimate gender from first name using common patterns.
    
    Note: This is a simplified approach and has significant limitations.
    Gender determination from names is inherently imprecise and culturally biased.
    This should only be used for aggregate analysis with appropriate caveats.
    """
    # Strip any titles, periods, or extra spaces
    if not first_name or first_name == "Unknown" or len(first_name) < 2:
        return "Unknown"
        
    first_name = first_name.lower().strip()
    
    # Common female name endings in various languages
    female_patterns = ['a', 'ie', 'ette', 'elle', 'ina', 'ia', 'lyn', 'en', 'ey', 'anne', 'enne']
    # Names that end with these are typically feminine in many languages
    
    # Common male name endings in various languages
    male_patterns = ['o', 'us', 'er', 'on', 'in', 'im', 'el', 'an', 'or', 'en', 'as']
    # Names that end with these are typically masculine in many languages
    
    # Simple very common first names (this is just a small sample)
    common_female_names = {'mary', 'jennifer', 'elizabeth', 'susan', 'margaret', 'sarah', 'lisa', 'emma', 'olivia',
                          'sophia', 'mia', 'anna', 'maria', 'elena', 'julia', 'laura', 'natalia', 'alice', 'helen'}
    
    common_male_names = {'john', 'robert', 'michael', 'william', 'david', 'richard', 'joseph', 'thomas', 'james',
                        'daniel', 'matthew', 'alexander', 'peter', 'paul', 'mark', 'andrew', 'george', 'henry'}
    
    # Check common name lists first
    if first_name in common_female_names:
        return "Female"
    if first_name in common_male_names:
        return "Male"
    
    # Check endings
    for pattern in female_patterns:
        if first_name.endswith(pattern):
            return "Female"
    
    for pattern in male_patterns:
        if first_name.endswith(pattern):
            return "Male"
    
    # If we can't determine, return Unknown
    return "Unknown"

def analyze_gender_diversity(conference_data):
    """
    Analyze gender diversity in QM conferences.
    
    Returns dictionaries with gender counts by year and talk type.
    """
    print("Analyzing gender diversity...")
    
    # Extract years, excluding 2025 if present
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    
    # Track gender by year and talk type
    gender_by_year = {year: {'Male': 0, 'Female': 0, 'Unknown': 0} for year in years}
    gender_by_talk_type = {'plenary_talks': {'Male': 0, 'Female': 0, 'Unknown': 0},
                          'parallel_talks': {'Male': 0, 'Female': 0, 'Unknown': 0},
                          'poster_talks': {'Male': 0, 'Female': 0, 'Unknown': 0}}
    
    for year in years:
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                for talk in conference_data[year][talk_type]:
                    # Get speaker/first author name
                    speaker_name = talk.get('Speaker', '')
                    if not speaker_name or speaker_name == 'Unknown':
                        first_author = talk.get('Authors', ['Unknown'])[0]
                        speaker_name = first_author
                    
                    # Extract first name - assume format is "Last, First" or "First Last"
                    first_name = "Unknown"
                    if ',' in speaker_name:
                        parts = speaker_name.split(',')
                        if len(parts) > 1:
                            first_name = parts[1].strip().split()[0]
                    else:
                        parts = speaker_name.split()
                        if len(parts) > 0:
                            first_name = parts[0].strip()
                    
                    # Estimate gender
                    gender = estimate_gender_from_name(first_name)
                    
                    # Update counters
                    gender_by_year[year][gender] += 1
                    gender_by_talk_type[talk_type][gender] += 1
    
    return gender_by_year, gender_by_talk_type

def create_gender_diversity_plot(gender_by_year, gender_by_talk_type):
    """
    Create visualizations for gender diversity analysis.
    
    Creates two plots:
    1. Gender distribution by year
    2. Gender distribution by talk type
    """
    print("Creating gender diversity visualizations...")
    
    # Extract years and calculate percentages for plotting
    years = sorted(gender_by_year.keys())
    
    # Setup figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Gender by Year
    male_pct = []
    female_pct = []
    unknown_pct = []
    
    for year in years:
        total = sum(gender_by_year[year].values())
        if total > 0:
            male_pct.append(gender_by_year[year]['Male'] / total * 100)
            female_pct.append(gender_by_year[year]['Female'] / total * 100)
            unknown_pct.append(gender_by_year[year]['Unknown'] / total * 100)
        else:
            male_pct.append(0)
            female_pct.append(0)
            unknown_pct.append(0)
    
    # Create stacked bar chart
    bar_width = 0.7
    ind = np.arange(len(years))
    
    ax1.bar(ind, male_pct, bar_width, label='Male', color='skyblue')
    ax1.bar(ind, female_pct, bar_width, bottom=male_pct, label='Female', color='pink')
    ax1.bar(ind, unknown_pct, bar_width, bottom=np.array(male_pct) + np.array(female_pct), 
           label='Unknown', color='lightgray')
    
    # Add labels and title
    ax1.set_xlabel('Conference Year')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Gender Distribution by Year')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(years, rotation=45)
    ax1.legend(loc='upper right')
    
    # Add percentages on the bars
    for i, year in enumerate(years):
        total = sum(gender_by_year[year].values())
        if total > 0:
            female_count = gender_by_year[year]['Female']
            female_percent = female_count / total * 100
            if female_percent > 5:  # Only add text if there's enough space
                ax1.text(i, male_pct[i] + female_pct[i]/2, f"{female_percent:.1f}%", 
                        ha='center', va='center', fontweight='bold')
    
    # Plot 2: Gender by Talk Type
    talk_types = ['plenary_talks', 'parallel_talks', 'poster_talks']
    talk_type_labels = ['Plenary', 'Parallel', 'Poster']
    
    male_counts = [gender_by_talk_type[tt]['Male'] for tt in talk_types]
    female_counts = [gender_by_talk_type[tt]['Female'] for tt in talk_types]
    unknown_counts = [gender_by_talk_type[tt]['Unknown'] for tt in talk_types]
    
    # Calculate percentages
    totals = [sum(gender_by_talk_type[tt].values()) for tt in talk_types]
    male_pct = [male_counts[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(talk_types))]
    female_pct = [female_counts[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(talk_types))]
    unknown_pct = [unknown_counts[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(talk_types))]
    
    # Create stacked bar chart
    ind = np.arange(len(talk_types))
    
    ax2.bar(ind, male_pct, bar_width, label='Male', color='skyblue')
    ax2.bar(ind, female_pct, bar_width, bottom=male_pct, label='Female', color='pink')
    ax2.bar(ind, unknown_pct, bar_width, bottom=np.array(male_pct) + np.array(female_pct), 
           label='Unknown', color='lightgray')
    
    # Add labels and title
    ax2.set_xlabel('Talk Type')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Gender Distribution by Talk Type')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(talk_type_labels)
    ax2.legend(loc='upper right')
    
    # Add female percentages on the bars
    for i, tt in enumerate(talk_types):
        if totals[i] > 0:
            female_percent = female_counts[i] / totals[i] * 100
            if female_percent > 5:  # Only add text if there's enough space
                ax2.text(i, male_pct[i] + female_pct[i]/2, f"{female_percent:.1f}%", 
                        ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/gender_diversity.pdf')
    plt.close()
    
    # Create a second figure showing female representation in plenary vs other talk types
    plt.figure(figsize=(10, 6))
    
    # Calculate female representation ratio
    plenary_female_pct = female_pct[0]
    other_female_pct = (female_counts[1] + female_counts[2]) / (totals[1] + totals[2]) * 100 if (totals[1] + totals[2]) > 0 else 0
    
    comparison_data = [plenary_female_pct, other_female_pct]
    
    # Create bar chart
    plt.bar(['Plenary Talks', 'Other Talks'], comparison_data, color=['darkred', 'navy'])
    plt.axhline(y=other_female_pct, color='black', linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.ylabel('Female Representation (%)')
    plt.title('Female Representation: Plenary vs Other Talks')
    
    # Add percentage labels
    for i, value in enumerate(comparison_data):
        plt.text(i, value/2, f"{value:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/gender_representation_comparison.pdf')
    plt.close()

    # Create a note about methodology limitations
    with open('figures/gender_analysis_note.txt', 'w') as f:
        f.write("IMPORTANT NOTE ON GENDER ANALYSIS METHODOLOGY\n\n")
        f.write("The gender analysis presented in these visualizations is based on algorithmic inference from first names.\n")
        f.write("This approach has significant limitations and inherent biases, including:\n\n")
        f.write("1. Names do not always reliably indicate gender identity\n")
        f.write("2. The algorithm uses simplified Western/European naming patterns\n")
        f.write("3. Many names are culturally specific and may be misclassified\n")
        f.write("4. Non-binary and other gender identities are not represented\n\n")
        f.write("These results should be interpreted as approximate patterns only and not definitive analyses.\n")
        f.write("A more accurate approach would require self-reported gender information which is not available in the dataset.\n")

# Add proper entry point at the end of the file
if __name__ == "__main__":
    analyze_conference_data()

