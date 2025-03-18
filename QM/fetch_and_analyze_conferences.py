import requests
import json
import re
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

# Increase all font sizes by 30% - handling both numeric and string font sizes
default_font_size = plt.rcParams.get('font.size', 10)
if isinstance(default_font_size, str):
    try:
        default_font_size = float(default_font_size)
    except ValueError:
        default_font_size = 10

# Set the base font size with 30% increase
new_font_size = default_font_size * 1.3

# Update all font-related parameters
plt.rcParams.update({
    'font.size': new_font_size,
    'axes.titlesize': 'large',  # Use relative size names instead of multiplication
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
os.makedirs('figs', exist_ok=True)

# Common country names and abbreviations
COUNTRY_KEYWORDS = {
    'USA': ['USA', 'United States', 'America', 'US'],
    'UK': ['UK', 'United Kingdom', 'Britain', 'England'],
    'Germany': ['Germany', 'DE', 'Deutschland'],
    'France': ['France', 'FR'],
    'Italy': ['Italy', 'IT', 'Italia'],
    'Japan': ['Japan', 'JP'],
    'China': ['China', 'CN'],
    'Korea': ['Korea', 'KR'],
    'Switzerland': ['Switzerland', 'CH', 'CERN'],
    'India': ['India', 'IN'],
    'Brazil': ['Brazil', 'BR'],
    'Russia': ['Russia', 'RU'],
    'Poland': ['Poland', 'PL'],
    'Netherlands': ['Netherlands', 'NL'],
    'Spain': ['Spain', 'ES'],
    'Canada': ['Canada', 'CA'],
    'Mexico': ['Mexico', 'MX'],
    'Australia': ['Australia', 'AU'],
}

# Institution to country mapping database
INSTITUTION_COUNTRY = {
    'MIT': 'USA',
    'CERN': 'Switzerland',
    'Berkeley': 'USA',
    'Brookhaven': 'USA',
    'BNL': 'USA',
    'FNAL': 'USA',
    'Fermilab': 'USA',
    'DESY': 'Germany',
    'KEK': 'Japan',
    'IHEP': 'China',
    'JINR': 'Russia',
    'RAL': 'UK',
    'INFN': 'Italy',
    'CEA': 'France',
    'GSI': 'Germany',
    'TRIUMF': 'Canada',
    'SLAC': 'USA',
    'Los Alamos': 'USA',
    'LANL': 'USA',
    'Oak Ridge': 'USA',
    'ORNL': 'USA',
    'PSI': 'Switzerland',
    'RIKEN': 'Japan',
    # Add more institutions as needed
}

# Update the conference locations with correct information
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
    '2025': 'Frankfurt, Germany'  # Updated to Frankfurt
}

# Manual flash talk counts from conference timetables
FLASH_TALK_COUNTS = {
    '2011': 8,   # From timetable
    '2012': 8,   # From session https://indico.cern.ch/event/181055/sessions/25214/
    '2014': 8,   # From timetable only
    '2015': 8,   # From timetable only
    '2017': 8,   # From Flash talks session
    '2018': 10,  # From Plenary Talk Best-poster flash talks
    '2019': 6,   # From timetable only
    '2022': 10,  # From Flash Talks session
    '2023': 10   # From session https://indico.cern.ch/event/1139644/sessions/488508/
}

def normalize_institute_name(name):
    """Normalize institute name for better matching"""
    if not name:
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters and extra whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove common words that don't help with matching
    for word in ['university', 'institute', 'national', 'laboratory', 'department', 
                'center', 'centre', 'research', 'of', 'for', 'and', 'the', 'in']:
        name = re.sub(r'\b' + word + r'\b', '', name)
    
    # Remove country codes in parentheses
    name = re.sub(r'\([a-z]{2}\)', '', name)
    
    # Remove numbers
    name = re.sub(r'\d+', '', name)
    
    # Remove extra spaces created during the process
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def load_institute_country_database():
    """Load institute-to-country mappings from external database file"""
    database_file = 'institute_country_database.csv'
    institute_country = {}
    normalized_map = {}
    
    # Add mappings for the specific remaining unknown institutes
    exact_mappings = {
        'B': 'Unknown',
        'CEA, Paris-Saclay University': 'France',
        'CEA-Saclay': 'France',
        'Central China Normal University': 'China',
        'Central China Normal University ': 'China',
        'Central China Normal University / Tsinghua University': 'China',
        'Central China Normal University, China': 'China',
        'Central China Normal University.': 'China',
        'D': 'Unknown',
        'EMMI/GSI': 'Germany',
        'F': 'Unknown',
        'Gesellschaft fuer Schwerionenforschung mbH (GSI)': 'Germany',
        'High Energy Accelerator Research Organization (KEK)': 'Japan',
        'I': 'Unknown',
        'INT, University of Washington': 'USA',
        'L': 'Unknown',
        'LBNL': 'USA',
        'N': 'Unknown',
        'PhD student': 'Unknown',
        'R': 'Unknown',
        'Research Division and ExtreMe Matter Institute EMMI, GSI Helmholtzzentrum für Schwerionenforschung, Darmstadt, Germany': 'Germany',
        'SINAP/LBNL': 'China',
        'STAR Collaboration': 'USA',
        'SUBATECH': 'France',
        'SUBATECH Nantes': 'France',
        'SUBATECH, Nantes': 'France',
        'SUNY, Stony Brook': 'USA',
        'State University of New York at Stony Brook': 'USA',
        'Stony Brook U./BNL': 'USA',
        'Stony Brook University': 'USA',
        'Stony Brook University and BNL': 'USA',
        'Stony Brook and BNL': 'USA',
        'Subatech': 'France',
        'Tsinghua University': 'China',
        'U': 'Unknown',
        'Uiniversity of california, Los Angeles': 'USA',
        'University of California - Los Angeles': 'USA',
        'University of California Los Angeles': 'USA',
        'University of California, Davis': 'USA',
        'University of California, Los Angeles': 'USA',
        'University of California, Riverside': 'USA',
        'University of Colorado Boulder': 'USA',
        'University of Colorado, Boulder': 'USA',
        'University of Maryland': 'USA',
        'University of Maryland, College Park': 'USA',
        'University of Minnesota': 'USA',
        'University of Tennessee, Knoxville': 'USA',
        'University of Washington': 'USA',
        'Unknown': 'Unknown',
        'Unknown-Unknown-Unknown': 'Unknown',
        'V': 'Unknown',
        'VECC': 'India',
        'Vanderbilt University': 'USA',
        'Variable Energy Cyclotron Centre': 'India',
        'Variable Energy Cyclotron Centre, Kolkata': 'India',
        'W': 'Unknown',
        'Wayne State University': 'USA',
        'Wayne state university': 'USA',
        'Yale University': 'USA',
        'Yale University-Unknown-Unknown': 'USA',
        'for the STAR collaboration': 'USA',
        'l': 'Unknown',
        'lbnl': 'USA',
        'stony brook university': 'USA',
        'subatech': 'France'
    }
    
    try:
        with open(database_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        institute = parts[0].strip()
                        country = parts[1].strip()
                        institute_country[institute] = country
                        
                        # Also store normalized version for fuzzy matching
                        normalized = normalize_institute_name(institute)
                        if normalized:
                            normalized_map[normalized] = country
        
        # Add the exact mappings for remaining institutes
        for institute, country in exact_mappings.items():
            institute_country[institute] = country
            normalized = normalize_institute_name(institute)
            if normalized:
                normalized_map[normalized] = country
                
        print(f"Loaded {len(institute_country)} institute-to-country mappings from database")
    except FileNotFoundError:
        print(f"Warning: Institute-country database file '{database_file}' not found")
        institute_country = exact_mappings
        normalized_map = {normalize_institute_name(k): v for k, v in exact_mappings.items() if normalize_institute_name(k)}
    
    return institute_country, normalized_map

def update_institute_country_database(unknown_institutes):
    """Update the database with unknown institutes that need mapping"""
    database_file = 'institute_country_database.csv'
    unknown_file = 'unknown_institutes.txt'
    
    # Write unknown institutes to a separate file for manual processing
    with open(unknown_file, 'w', encoding='utf-8') as f:
        f.write("# Unknown institutes that need country mapping\n")
        f.write("# Format: Institute,Country\n")
        for institute in sorted(unknown_institutes):
            f.write(f"{institute},\n")
    
    print(f"Wrote {len(unknown_institutes)} unknown institutes to '{unknown_file}'")
    print("Please add country information to these institutes and merge into the main database")

def extract_country(affiliation, institute_country_db):
    """Extract country from affiliation using various methods including database lookup"""
    if not affiliation:
        return 'Unknown'
    
    # First try to find country code in parentheses at the end: "Something (US)"
    parentheses_match = re.search(r'\(([A-Z]{2})\)$', affiliation)
    if parentheses_match:
        country_code = parentheses_match.group(1)
        # Map common country codes to full names
        country_code_map = {
            'US': 'USA',
            'UK': 'UK',
            'DE': 'Germany',
            'FR': 'France',
            'IT': 'Italy',
            'JP': 'Japan',
            'CN': 'China',
            'KR': 'Korea',
            'CH': 'Switzerland',
            'IN': 'India',
            # Add more mappings as needed
        }
        return country_code_map.get(country_code, country_code)
    
    # Check against institute-country database
    for inst, country in institute_country_db.items():
        if inst.upper() in affiliation.upper():
            return country
    
    # Try to find country keywords
    affiliation_upper = affiliation.upper()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(keyword.upper() in affiliation_upper for keyword in keywords):
            return country
    
    # If no country is found, return the last part of the affiliation
    parts = [p.strip() for p in affiliation.split(',')]
    return parts[-1] if parts else 'Unknown'

def validate_indico_url(indico_id, year):
    """Validate Indico URL and check if it's the correct conference"""
    url = f"https://indico.cern.ch/export/event/{indico_id}.json?detail=contributions&pretty=yes"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if not data or 'results' not in data or not data['results']:
            return False, "No data found", None
            
        event_title = data['results'][0].get('title', '').lower()
        print(f"\nChecking URL: {url}")
        print(f"Response status code: {response.status_code}")
        print(f"Event title: {data['results'][0].get('title', '')}")
        
        # Special case for QM2022
        if year == '2022' and indico_id == '895086':
            return True, "Valid Indico page with correct year", data
            
        # Original validation logic from before
        if year == '2011':
            if 'qm 2011' in event_title or 'xxii international conference' in event_title:
                return True, "Valid Indico page with correct year", data
        
        valid_titles = [
            f'quark matter {year}',
            f'qm {year}',
            f'qm{year}',
            'quark matter',
            'qm'
        ]
        
        if any(title in event_title for title in valid_titles):
            return True, "Valid Indico page with correct year", data
            
        return False, f"Title mismatch: {data['results'][0].get('title', '')}", None
        
    except requests.exceptions.RequestException as e:
        return False, f"Error fetching URL: {str(e)}", None
    except ValueError as e:
        return False, f"Error parsing JSON: {str(e)}", None

def categorize_session(session_name, title, year):
    """Categorize a session as plenary, parallel, or poster"""
    # Convert inputs to strings and handle None values
    session_lower = str(session_name or '').lower()
    title_lower = str(title or '').lower()
    
    # Special handling for 2011
    if year == '2011':
        if isinstance(session_name, dict):
            session_lower = str(session_name.get('title', '')).lower()
        if 'plenary' in session_lower:
            return "plenary"
        if any(x in session_lower for x in ['parallel', 'track']):
            return "parallel"
        if 'poster' in session_lower:
            return "poster"
    
    # Regular categorization for other years
    if 'poster' in session_lower or 'poster' in title_lower:
        return "poster"
    
    # Check for other non-talk sessions
    if any(x in session_lower or x in title_lower for x in [
        'flash', 'student day', 'teacher', 'award', 'medal',
        'opening', 'closing', 'welcome'
    ]):
        return "other"
    
    # Only exclude discussions from plenary, but keep them for parallel
    if 'discussion' in title_lower and 'plenary' in session_lower:
        return "other"
    
    # Year-specific patterns
    if year == '2023':
        if 'plenary session' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'jets', 'heavy flavor', 'collective dynamics', 'new theory',
            'small systems', 'initial state', 'qcd at finite t',
            'light flavor', 'em probes', 'critical point', 'chirality',
            'spin/eic physics', 'future experiments', 'astrophysics', 'upc',
            'discussion'  # Include discussions in parallel sessions
        ]):
            return "parallel"
            
    elif year == '2014':
        if 'plenary' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'heavy flavor', 'jets', 'correlations and fluctuations',
            'collective dynamics', 'qcd phase diagram', 'electromagnetic probes',
            'initial state physics', 'new theoretical developments',
            'thermodynamics and hadron chemistry', 'approach to equilibrium',
            'discussion'  # Include discussions in parallel sessions
        ]):
            return "parallel"
            
    elif year == '2015':
        if 'plenary session' in session_lower:
            return "plenary"
        if any(x in session_lower.replace('-', ' ') for x in [
            'jets and high pt', 'correlations and fluctuations',
            'qgp in small systems', 'initial state physics',
            'open heavy flavors', 'collective dynamics',
            'quarkonia', 'electromagnetic probes',
            'discussion'  # Include discussions in parallel sessions
        ]):
            return "parallel"
            
    elif year == '2018':
        if 'plenary' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'jet modifications', 'collective dynamics', 'collectivity in small systems',
            'quarkonia', 'initial state physics', 'correlations and fluctuations',
            'open heavy flavour', 'chirality', 'phase diagram', 'qcd at high temperature',
            'electromagnetic and weak probes', 'new theoretical developments',
            'thermodynamics and hadron chemistry', 'high baryon density',
            'discussion'  # Include discussions in parallel sessions
        ]):
            return "parallel"
            
    else:
        # General patterns for other years
        if 'plenary' in session_lower:
            return "plenary"
        if 'parallel' in session_lower or 'discussion' in session_lower:
            return "parallel"
    
    return "unknown"

def should_exclude_contribution(title, session, year):
    """Check if a contribution should be excluded from statistics"""
    title_lower = str(title).lower()
    session_lower = str(session).lower()
    
    # Basic exclusion keywords for all years
    exclude_keywords = [
        'qm2021',
        'awards',
        'closing',
        'opening',
        'welcome',
        'medal',
        'flash talks',
        'zimanyi',
        'theory medal',
        'presentation',
        'ceremony'
    ]
    
    # Year-specific exclusions
    if year == '2015':
        if 'round table discussion' in title_lower:
            return True
        if 'special session' in session_lower:
            return True
        if session_lower == 'correlations and fluctuations ii':
            return True
    
    # Check both title and session name for general exclusion keywords
    return any(keyword in title_lower or keyword in session_lower 
              for keyword in exclude_keywords)

def extract_speaker_info(speakers):
    """Extract speaker information from raw data"""
    if not speakers:
        return "No name", "Unknown", "Unknown"
        
    speaker = speakers[0]
    
    # Try different name formats
    name = (speaker.get('fullName') or 
            f"{speaker.get('first_name', '')} {speaker.get('last_name', '')}" or 
            speaker.get('name', 'No name')).strip()
    
    affiliation = speaker.get('affiliation', 'Unknown')
    country = speaker.get('country', 'Unknown')
    
    # Special cases for known speakers
    if name == "Hatsuda, Tetsuo":
        affiliation = "RIKEN"
        country = "Japan"
    
    return name, affiliation, country

# Add manual corrections for known cases
MANUAL_CORRECTIONS = {
    '2015': {
        'Systematics of higher order net-baryon number fluctuations at small values of the baryon chemical potential: A comparison of lattice QCD and beam energy scan results': {
            'Speaker': 'Karsch, Frithjof',
            'Institute': 'Brookhaven National Laboratory & Bielefeld University',
            'Country': 'USA & Germany'
        }
    },
    '2018': {
        'A novel invariant mass method to isolate resonance backgrounds from the chiral magnetic effect': {
            'Speaker': 'Wang, Fuqiang',
            'Institute': 'Purdue University & Huzhou University',
            'Country': 'USA & China'
        }
    }
}

def apply_manual_corrections(talk_data, year):
    """Apply manual corrections for known cases"""
    if year in MANUAL_CORRECTIONS:
        title = talk_data['Title']
        if title in MANUAL_CORRECTIONS[year]:
            corrections = MANUAL_CORRECTIONS[year][title]
            talk_data.update(corrections)
    return talk_data

def fetch_and_process_contributions(indico_id, year):
    """Fetch and process contributions from Indico"""
    is_valid, message, data = validate_indico_url(indico_id, year)
    
    if not is_valid or not data:
        return None
    
    try:
        results = data.get('results', [])
        if not results:
            return None
            
        event_data = results[0]
        contributions = event_data.get('contributions', [])
        
        all_talks = []
        plenary_talks = []
        parallel_talks = []
        poster_talks = []
        flash_talks = []
        other_talks = []
        unknown_plenary = []
        unknown_parallel = []
        
        # Common keywords for all years
        sunday_keywords = ['sunday', 'Sunday']
        
        # Manual corrections for 2011
        manual_corrections_2011 = {
            'Satow, Daisuke': {'Institute': 'RIKEN', 'Country': 'Japan'}
        }
        
        if year == '2011':
            print(f"\nProcessing contributions for QM2011...")
            
            # Define other sessions for 2011
            other_sessions = ['Famous plot session']
            
            for i, contribution in enumerate(contributions, 1):
                title = contribution.get('title', '')
                contrib_type = contribution.get('type', '')
                track = contribution.get('track', '')
                
                # Extract speaker information
                speakers = (
                    contribution.get('speakers', []) or 
                    contribution.get('person_links', []) or 
                    contribution.get('primary_authors', []) or 
                    contribution.get('coauthors', [])
                )
                
                if not speakers and contribution.get('primaryauthors'):
                    speakers = contribution['primaryauthors']
                
                name, affiliation, country = extract_speaker_info(speakers)
                
                # Apply manual corrections for 2011
                if year == '2011' and name in manual_corrections_2011:
                    correction = manual_corrections_2011[name]
                    affiliation = correction['Institute']
                    country = correction['Country']
                
                # Common check for Sunday sessions first
                if any(sunday in str(title) for sunday in sunday_keywords) or \
                   any(sunday in str(track) for sunday in sunday_keywords):
                    session_type = 'other'
                # Then 2011-specific categorization
                elif track in other_sessions:
                    session_type = 'other'
                elif contrib_type == 'Poster':
                    session_type = 'poster'
                elif contrib_type == 'Plenary':
                    session_type = 'plenary'
                elif contrib_type == 'Parallel':
                    session_type = 'parallel'
                elif contrib_type == 'Flash':
                    session_type = 'flash'
                elif 'plenary' in str(track).lower():
                    session_type = 'unknown_plenary'
                elif track:
                    session_type = 'unknown_parallel'
                else:
                    session_type = 'unknown_parallel'
                
                talk_data = {
                    'Session': track,
                    'Type': session_type,
                    'Title': title,
                    'Speaker': name,
                    'Institute': affiliation,
                    'Country': country,
                    'Raw_Speaker_Data': speakers[0] if speakers else None
                }
                
                all_talks.append(talk_data)
                if session_type == "plenary":
                    plenary_talks.append(talk_data)
                elif session_type == "parallel":
                    parallel_talks.append(talk_data)
                elif session_type == "poster":
                    poster_talks.append(talk_data)
                elif session_type == "flash":
                    flash_talks.append(talk_data)
                elif session_type == "other":
                    other_talks.append(talk_data)
                elif session_type == "unknown_plenary":
                    unknown_plenary.append(talk_data)
                elif session_type == "unknown_parallel":
                    unknown_parallel.append(talk_data)
        
        elif year == '2025':
            # Define session categories
            other_sessions = ['Early Career Researcher Day']
            
            for i, contribution in enumerate(contributions, 1):
                title = contribution.get('title', '')
                session = contribution.get('session', '')
                contrib_type = str(contribution.get('type', '')).lower()
                session_str = str(session)
                
                # Add Sunday check before other categorization
                if any(sunday in str(title) for sunday in sunday_keywords) or \
                   any(sunday in str(session) for sunday in sunday_keywords):
                    session_type = 'other'
                elif 'poster' in session_str.lower() or 'poster' in contrib_type:
                    session_type = 'poster'
                elif any(os in session_str for os in other_sessions):
                    session_type = 'other'
                else:
                    session_type = categorize_session(session, title, year)
                
                # Extract speaker information
                speakers = (contribution.get('speakers', []) or 
                          contribution.get('person_links', []) or 
                          contribution.get('primary_authors', []))
                
                name, affiliation, country = extract_speaker_info(speakers)
                
                talk_data = {
                    'Session': session,
                    'Type': session_type,
                    'Title': title,
                    'Speaker': name,
                    'Institute': affiliation,
                    'Country': country,
                    'Raw_Speaker_Data': speakers[0] if speakers else None
                }
                
                all_talks.append(talk_data)
                if session_type == "plenary":
                    plenary_talks.append(talk_data)
                elif session_type == "parallel":
                    parallel_talks.append(talk_data)
                elif session_type == "poster":
                    poster_talks.append(talk_data)
                elif session_type == "other":
                    other_talks.append(talk_data)
        
        else:
            # Normal processing for other years
            for contribution in contributions:
                # Add Sunday check before normal categorization
                title = contribution.get('title', '')
                session = contribution.get('session', '')
                
                if any(sunday in str(title) for sunday in sunday_keywords) or \
                   any(sunday in str(session) for sunday in sunday_keywords):
                    session_type = 'other'
                else:
                    session_type = categorize_session(session, title, year)
                
                # Extract speaker information
                speakers = (contribution.get('speakers', []) or 
                          contribution.get('person_links', []) or 
                          contribution.get('primary_authors', []))
                
                name, affiliation, country = extract_speaker_info(speakers)
                
                talk_data = {
                    'Session': session,
                    'Type': session_type,
                    'Title': title,
                    'Speaker': name,
                    'Institute': affiliation,
                    'Country': country,
                    'Raw_Speaker_Data': speakers[0] if speakers else None
                }
                
                # Apply any manual corrections
                talk_data = apply_manual_corrections(talk_data, year)
                
                all_talks.append(talk_data)
                if session_type == "plenary":
                    plenary_talks.append(talk_data)
                elif session_type == "parallel":
                    parallel_talks.append(talk_data)
                elif session_type == "poster":
                    poster_talks.append(talk_data)
        
        # Calculate totals and unknown affiliations for each category
        total_main = len(plenary_talks) + len(parallel_talks) + len(poster_talks) + len(flash_talks)
        
        unknown_plenary_aff = sum(1 for talk in plenary_talks if not talk['Institute'])
        unknown_parallel_aff = sum(1 for talk in parallel_talks if not talk['Institute'])
        unknown_poster_aff = sum(1 for talk in poster_talks if not talk['Institute'])
        unknown_flash_aff = sum(1 for talk in flash_talks if not talk['Institute'])
        total_unknown_aff = unknown_plenary_aff + unknown_parallel_aff + unknown_poster_aff + unknown_flash_aff
        
        if year == '2011':
            print(f"\nFinished processing QM2011:")
            print(f"Total (main categories): {total_main}")
            print(f"\nDetailed breakdown:")
            print(f"Plenary: {len(plenary_talks)} (Unknown aff: {unknown_plenary_aff})")
            print(f"Parallel: {len(parallel_talks)} (Unknown aff: {unknown_parallel_aff})")
            print(f"Poster: {len(poster_talks)} (Unknown aff: {unknown_poster_aff})")
            print(f"Flash: {len(flash_talks)} (Unknown aff: {unknown_flash_aff})")
            print(f"\nTotal unknown affiliations: {total_unknown_aff}")
            
            # Print details of unknown affiliations
            print("\nDetails of unknown affiliations:")
            
            if unknown_plenary_aff > 0:
                print("\nPlenary talks with unknown affiliations:")
                for talk in plenary_talks:
                    if not talk['Institute']:
                        print(f"- {talk['Title']} (Speaker: {talk['Speaker']})")
            
            if unknown_parallel_aff > 0:
                print("\nParallel talks with unknown affiliations:")
                for talk in parallel_talks:
                    if not talk['Institute']:
                        print(f"- {talk['Title']} (Speaker: {talk['Speaker']})")
            
            if unknown_poster_aff > 0:
                print("\nPoster talks with unknown affiliations:")
                for talk in poster_talks:
                    if not talk['Institute']:
                        print(f"- {talk['Title']} (Speaker: {talk['Speaker']})")
            
            if unknown_flash_aff > 0:
                print("\nFlash talks with unknown affiliations:")
                for talk in flash_talks:
                    if not talk['Institute']:
                        print(f"- {talk['Title']} (Speaker: {talk['Speaker']})")
            
            print(f"\nOther categories (not in total):")
            print(f"Other: {len(other_talks)}")
            print(f"Unknown Plenary: {len(unknown_plenary)}")
            print(f"Unknown Parallel: {len(unknown_parallel)}")
        
        return {
            'all_talks': all_talks,
            'plenary_talks': plenary_talks,
            'parallel_talks': parallel_talks,
            'poster_talks': poster_talks,
            'flash_talks': flash_talks,
            'other_talks': other_talks,
            'unknown_plenary': unknown_plenary,
            'unknown_parallel': unknown_parallel,
            'total_main': total_main,
            'unknown_affiliations': {
                'plenary': unknown_plenary_aff,
                'parallel': unknown_parallel_aff,
                'poster': unknown_poster_aff,
                'flash': unknown_flash_aff,
                'total': total_unknown_aff
            }
        }
        
    except Exception as e:
        print(f"Error processing QM{year}: {str(e)}")
        return None

def plot_distributions(all_data, plenary_data, parallel_data, year, verbose=True):
    # Create directory for this conference year
    os.makedirs(f'figs/QM{year}', exist_ok=True)
    
    # Count countries and institutes for analysis
    country_counts = Counter([talk['Country'] for talk in all_data])
    institute_counts = Counter([talk['Institute'] for talk in all_data])
    
    # Only print detailed statistics if verbose mode is on
    if verbose:
        print(f"\nQM{year} Statistics:")
        print("Total number of talks:", len(all_data))
        print("\nTop 10 countries by number of talks:")
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{country}: {count}")
        print("\nTop 10 institutes by number of talks:")
        for inst, count in sorted(institute_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{inst}: {count}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: All talks per country
    plt.subplot(3, 2, 1)
    plt.bar(country_counts.keys(), country_counts.values())
    plt.title(f'QM{year}: All Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 2: All talks per institute (top 20)
    plt.subplot(3, 2, 2)
    top_20_institutes = dict(sorted(institute_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.bar(top_20_institutes.keys(), top_20_institutes.values())
    plt.title(f'QM{year}: All Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Plenary talks per country
    plt.subplot(3, 2, 3)
    plenary_country_counts = Counter([talk['Country'] for talk in plenary_data])
    plt.bar(plenary_country_counts.keys(), plenary_country_counts.values())
    plt.title(f'QM{year}: Plenary Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: Plenary talks per institute
    plt.subplot(3, 2, 4)
    plenary_institute_counts = Counter([talk['Institute'] for talk in plenary_data])
    plt.bar(plenary_institute_counts.keys(), plenary_institute_counts.values())
    plt.title(f'QM{year}: Plenary Talks by Institute')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 5: Parallel talks per country
    plt.subplot(3, 2, 5)
    parallel_country_counts = Counter([talk['Country'] for talk in parallel_data])
    plt.bar(parallel_country_counts.keys(), parallel_country_counts.values())
    plt.title(f'QM{year}: Parallel Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 6: Parallel talks per institute (top 20)
    plt.subplot(3, 2, 6)
    parallel_institute_counts = Counter([talk['Institute'] for talk in parallel_data])
    top_20_parallel = dict(sorted(parallel_institute_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.bar(top_20_parallel.keys(), top_20_parallel.values())
    plt.title(f'QM{year}: Parallel Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'figs/QM{year}/talk_distributions.pdf', bbox_inches='tight')
    
    # Return data for possible cross-conference analysis
    return {
        'all_talks': len(all_data),
        'plenary_talks': len(plenary_data),
        'parallel_talks': len(parallel_data),
        'country_counts': country_counts,
        'institute_counts': institute_counts
    }

def analyze_trends_across_conferences(conference_data):
    """Analyze trends across multiple conferences"""
    if not conference_data:
        print("No conference data available for trend analysis")
        return
    
    # Create directory for trend analysis
    os.makedirs('figs/trends', exist_ok=True)
    
    # Extract years and sort them
    years = sorted(conference_data.keys())
    
    # Plot number of talks by year
    plt.figure(figsize=(12, 6))
    plt.plot(years, [conference_data[year]['all_talks'] for year in years], 'o-', label='All Talks')
    plt.plot(years, [conference_data[year]['plenary_talks'] for year in years], 's-', label='Plenary Talks')
    plt.plot(years, [conference_data[year]['parallel_talks'] for year in years], '^-', label='Parallel Talks')
    plt.title('Number of Talks by Conference Year')
    plt.xlabel('QM Year')
    plt.ylabel('Number of Talks')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/trends/talks_by_year.pdf', bbox_inches='tight')
    
    # Analyze top countries across years
    # Get all unique countries
    all_countries = set()
    for year in years:
        all_countries.update(conference_data[year]['country_counts'].keys())
    
    # Select top countries overall for tracking
    country_total = Counter()
    for year in years:
        country_total.update(conference_data[year]['country_counts'])
    
    top_countries = [country for country, _ in country_total.most_common(8)]
    
    # Plot trends for top countries
    plt.figure(figsize=(14, 8))
    for country in top_countries:
        country_by_year = [conference_data[year]['country_counts'].get(country, 0) for year in years]
        plt.plot(years, country_by_year, 'o-', label=country)
    
    plt.title('Contributions from Top Countries Over Time')
    plt.xlabel('QM Year')
    plt.ylabel('Number of Contributions')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/trends/country_trends.pdf', bbox_inches='tight')
    
    # Print trend summary
    print("\n=== Trend Analysis Across QM Conferences ===")
    print(f"Years analyzed: {', '.join(years)}")
    print("\nTotal talks by year:")
    for year in years:
        print(f"QM{year}: {conference_data[year]['all_talks']} talks")
    
    print("\nTop countries across all conferences:")
    for country, count in country_total.most_common(10):
        print(f"{country}: {count} talks")

def plot_conference_statistics(conference_data):
    """Plot conference statistics over the years"""
    # Prepare data
    years = []
    totals = []
    plenaries = []
    parallels = []
    posters = []
    flashes = []
    
    # Sort by year and collect data
    for year in sorted(conference_data.keys()):
        data = conference_data[year]
        years.append(int(year))
        totals.append(len(data['all_talks']))
        plenaries.append(len(data['plenary_talks']))
        parallels.append(len(data['parallel_talks']))
        posters.append(len(data['poster_talks']))
        flashes.append(FLASH_TALK_COUNTS.get(year, 0))  # Add manual flash talk counts
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot lines with markers
    plt.plot(years, totals, 'bo-', label='Total Contributions', linewidth=2, markersize=8)
    plt.plot(years, plenaries, 'rs-', label='Plenary Talks', linewidth=2, markersize=8)
    plt.plot(years, parallels, 'gv-', label='Parallel Talks', linewidth=2, markersize=8)
    plt.plot(years, posters, 'md-', label='Posters', linewidth=2, markersize=8)
    plt.plot(years, flashes, 'c^-', label='Flash Talks', linewidth=2, markersize=8)  # Add flash talks line
    
    # Customize the plot
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Talks', fontsize=12)
    plt.title('QM Conference Talk Statistics Over Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust axis
    plt.xticks(years, rotation=45)
    
    # Add value labels
    for x, y1, y2, y3, y4 in zip(years, totals, plenaries, parallels, posters):
        plt.text(x, y1+20, str(y1), ha='center', va='bottom')
        plt.text(x, y2-15, str(y2), ha='center', va='top')
        plt.text(x, y3+10, str(y3), ha='center', va='bottom')
        plt.text(x, y4+10, str(y4), ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save the plot as PDF in figures directory
    plt.savefig('figures/QM_talk_statistics.pdf')
    plt.close()

def extract_keywords(title):
    """Extract meaningful keywords from title"""
    # Common words to exclude
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                 'to', 'was', 'were', 'will', 'with', 'via', 'using', 'vs', 'versus',
                 'new', 'recent', 'results', 'measurements', 'study', 'studies',
                 'measurement', 'analysis', 'observation', 'observations', 'evidence',
                 'search', 'searching', 'towards', 'approach', 'investigation'}
    
    # Remove special characters and convert to lowercase
    title = title.lower()
    title = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in title)
    
    # Split into words and filter
    words = title.split()
    keywords = [word for word in words 
               if word not in stop_words 
               and len(word) > 2  # Skip very short words
               and not word.isdigit()]  # Skip pure numbers
    
    return keywords

def print_summary(conferences):
    """Print summary of all conferences"""
    print("\nSummary of all conferences:")
    print("Year  Location                  Total  Plenary  Parallel  Poster  Flash  Unknown_Aff  Unknown_Inst")
    print("-" * 85)
    
    # Prepare data for the plot
    years = []
    top_keywords = []
    top_counts = []
    
    # First, let's print the full structure to debug
    print("\nDebug - conferences type:", type(conferences))
    print("Debug - first item type:", type(conferences[0]) if conferences else "No conferences")
    print("Debug - first item keys:", conferences[0].keys() if conferences and isinstance(conferences[0], dict) else "Not a dict")
    
    # Sort conferences by year
    conferences_sorted = sorted(conferences, key=lambda x: x['year'] if isinstance(x, dict) and 'year' in x else '')
    
    for conf in conferences_sorted:
        # Check if conf is a dictionary
        if not isinstance(conf, dict):
            print(f"Debug - Skipping non-dict conf: {conf}")
            continue
            
        year = conf.get('year', '')
        location = conf.get('location', '')
        
        # Check if necessary data exists
        if not all(key in conf for key in ['plenary_talks', 'parallel_talks', 'poster_talks']):
            print(f"Debug - Missing talk data for {year}")
            continue
            
        plenary = len(conf.get('plenary_talks', []))
        parallel = len(conf.get('parallel_talks', []))
        poster = len(conf.get('poster_talks', []))
        flash = len(conf.get('flash_talks', []))
        total = plenary + parallel + poster + flash
        
        # Count unknown affiliations and institutes
        unknown_aff = conf.get('unknown_affiliations', {}).get('total', 0)
        all_talks = (conf.get('plenary_talks', []) + 
                   conf.get('parallel_talks', []) + 
                   conf.get('poster_talks', []) + 
                   conf.get('flash_talks', []))
        unknown_inst = len(set(talk.get('Institute', '') for talk in all_talks if not talk.get('Institute')))
        
        print(f"{year}  {location:<22} {total:>6}  {plenary:>7}  {parallel:>8}  {poster:>6}  {flash:>5}  {unknown_aff:>10}  {unknown_inst:>12}")
        
        try:
            # Analyze keywords
            keywords_count = {}
            for talk in all_talks:
                title = talk.get('Title', '')
                if title:
                    keywords = extract_keywords(title)
                    for keyword in keywords:
                        keywords_count[keyword] = keywords_count.get(keyword, 0) + 1
            
            # Get top 3 keywords
            sorted_keywords = sorted(keywords_count.items(), key=lambda x: x[1], reverse=True)
            if sorted_keywords:
                print(f"Top keywords: {', '.join([f'{k}({v})' for k, v in sorted_keywords[:3]])}")
                
                # Store data for plotting
                years.append(year)
                top3_keywords = [kw for kw, _ in sorted_keywords[:3]]
                top3_counts = [count for _, count in sorted_keywords[:3]]
                top_keywords.append(top3_keywords)
                top_counts.append(top3_counts)
        except Exception as e:
            print(f"Error analyzing keywords for {year}: {str(e)}")
    
    try:
        if years and top_keywords and top_counts:
            # Create figures directory if it doesn't exist
            figures_dir = 'figures'
            os.makedirs(figures_dir, exist_ok=True)
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            x = np.arange(len(years))
            width = 0.25
            
            # Plot bars for each keyword position
            for i in range(3):
                counts = [counts[i] if i < len(counts) else 0 for counts in top_counts]
                plt.bar(x + i*width, counts, width, 
                       label=f'#{i+1} keyword',
                       alpha=0.8)
            
            # Customize the plot
            plt.xlabel('Conference Year')
            plt.ylabel('Keyword Count')
            plt.title('Top 3 Keywords Usage in QM Conferences')
            plt.xticks(x + width, years)
            plt.legend()
            
            # Add keyword labels on top of bars
            for i in range(len(years)):
                for j in range(3):
                    if j < len(top_keywords[i]):
                        plt.text(i + j*width, top_counts[i][j], 
                                top_keywords[i][j],
                                ha='center', va='bottom',
                                rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            save_path = os.path.join(figures_dir, 'top_keywords.pdf')
            print(f"Saving plot to {save_path}")
            plt.savefig(save_path, format='pdf')
            plt.close()
            print("Plot saved successfully")
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

def create_keywords_plot(all_conference_data):
    """Create a comprehensive visualization of keyword trends across conferences"""
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from wordcloud import WordCloud
    import os
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Define important physics concepts to track over time
    physics_concepts = {
        'QGP & Plasma': ['qgp', 'plasma', 'quark gluon', 'quark-gluon'],
        'Flow & Collectivity': ['flow', 'collective', 'hydro', 'viscosity', 'viscous'],
        'Small Systems': ['small system', 'pp collision', 'p-p', 'p-pb', 'p-a'],
        'Fluctuations': ['fluctuation', 'correlations', 'correlation'],
        'Heavy Flavor': ['charm', 'bottom', 'heavy flavor', 'heavy quark', 'quarkonia', 'quarkonium', 'j/psi', 'jpsi', 'upsilon'],
        'Jets & Energy Loss': ['jet', 'energy loss', 'quenching', 'high pt', 'high-pt'],
        'Phase Diagram': ['phase', 'transition', 'critical point', 'qcd phase'],
        'EM Probes': ['photon', 'dilepton', 'electromagnetic', 'em probe'],
        'Initial State': ['initial state', 'glasma', 'cgc', 'color glass']
    }

    # Define facilities to track
    facilities = {
        'LHC': ['lhc', 'alice', 'cms', 'atlas'],
        'RHIC': ['rhic', 'star', 'phenix', 'brahms', 'phobos'],
        'Future': ['eic', 'nica', 'fair', 'j-parc', 'jparc']
    }
    
    # Function to count concept occurrences in titles
    def count_concepts(titles, concept_dict):
        counts = {concept: 0 for concept in concept_dict}
        
        for title in titles:
            if not isinstance(title, str):
                continue
            title_lower = title.lower()
            for concept, keywords in concept_dict.items():
                if any(keyword in title_lower for keyword in keywords):
                    counts[concept] += 1
        
        return counts
    
    # Prepare data for analysis
    years = sorted(all_conference_data.keys())
    concept_trends = {concept: [] for concept in physics_concepts}
    facility_trends = {facility: [] for facility in facilities}
    
    # Collect all titles for each year
    all_titles_by_year = {}
    for year in years:
        if year in all_conference_data:
            all_talks = all_conference_data[year]['all_talks']
            titles = [talk.get('Title', '') for talk in all_talks if talk.get('Title')]
            all_titles_by_year[year] = titles
            
            # Count physics concepts
            concept_counts = count_concepts(titles, physics_concepts)
            for concept, count in concept_counts.items():
                concept_trends[concept].append(count / len(titles) * 100 if titles else 0)
                
            # Count facility mentions
            facility_counts = count_concepts(titles, facilities)
            for facility, count in facility_counts.items():
                facility_trends[facility].append(count / len(titles) * 100 if titles else 0)
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Physics concepts subplot
    ax1 = plt.subplot(gs[0])
    for concept, percentages in concept_trends.items():
        ax1.plot(years, percentages, marker='o', linewidth=2, label=concept)
    
    ax1.set_title('Evolution of Physics Concepts in Quark Matter Conferences', fontsize=14)
    ax1.set_ylabel('Percentage of Presentations (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Facilities subplot
    ax2 = plt.subplot(gs[1])
    for facility, percentages in facility_trends.items():
        ax2.plot(years, percentages, marker='s', linewidth=2, label=facility)
    
    ax2.set_title('Mentions of Experimental Facilities', fontsize=14)
    ax2.set_xlabel('Conference Year', fontsize=12)
    ax2.set_ylabel('Percentage of Presentations (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig('figures/keyword_trends.pdf', bbox_inches='tight')
    plt.savefig('figures/keyword_trends.png', dpi=300, bbox_inches='tight')
    
    # Common stopwords to remove
    stopwords = set(['and', 'the', 'in', 'of', 'for', 'on', 'with', 'at', 'from', 'by', 
                    'to', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'as', 'if',
                    'then', 'else', 'when', 'up', 'down', 'conference', 'study', 'analysis',
                    'measurement', 'results', 'data', 'using', 'via', 'new', 'recent',
                    'quark', 'matter', 'qm', 'physics', 'collision', 'collisions', 'ion', 'ions',
                    'heavy', 'experiment', 'experimental', 'theory', 'theoretical', 'model', 'models'])
    
    # Combine both visualizations into one comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3)
    
    # Top row: Trend lines for physics concepts and facilities
    ax_concepts = plt.subplot(gs[0, :2])
    for concept, percentages in concept_trends.items():
        ax_concepts.plot(years, percentages, marker='o', linewidth=2, label=concept)
    ax_concepts.set_title('Evolution of Physics Concepts', fontsize=14)
    ax_concepts.set_ylabel('Percentage of Presentations (%)', fontsize=12)
    ax_concepts.grid(True, linestyle='--', alpha=0.7)
    ax_concepts.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    ax_facilities = plt.subplot(gs[0, 2])
    for facility, percentages in facility_trends.items():
        ax_facilities.plot(years, percentages, marker='s', linewidth=2, label=facility)
    ax_facilities.set_title('Experimental Facilities', fontsize=14)
    ax_facilities.grid(True, linestyle='--', alpha=0.7)
    ax_facilities.legend()
    
    # Custom colormap for word clouds
    colors = [(0.8, 0.3, 0.3), (0.3, 0.3, 0.8), (0.3, 0.8, 0.3)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Bottom two rows: Word clouds for each year
    row_col_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    
    for i, year in enumerate(years):
        if i >= len(row_col_positions):
            break  # Skip if we run out of positions
            
        row, col = row_col_positions[i]
        if year in all_titles_by_year:
            ax = plt.subplot(gs[row, col])
            titles = all_titles_by_year[year]
            
            if titles:
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     colormap=cmap,
                                     stopwords=stopwords,
                                     max_words=30,
                                     contour_width=1, contour_color='black').generate(' '.join(titles))
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'QM{year}', fontsize=14)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/QM_keyword_analysis.pdf', bbox_inches='tight')
    plt.savefig('figures/QM_keyword_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Keyword analysis completed and visualizations saved.")

def create_country_institute_plots(all_conference_data):
    """Create plots showing distribution of talks by country and institute"""
    print("\nCreating plots for country and institute distributions...")
    
    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Direct mappings for problematic institutes
    direct_mappings = {
        'B': 'Unknown',
        'D': 'Unknown',
        'Department of Physics and Technology': 'Norway',
        'F': 'Unknown',
        'Federal University of Alfenas': 'Brazil',
        'GSI Darmstadt': 'Germany',
        'GSI, Darmstadt': 'Germany',
        'GSI, Darmstadt, Germany': 'Germany',
        'GSI/NNRC': 'Germany',
        'I': 'Unknown',
        'L': 'Unknown',
        'LIP, Lisbon': 'Portugal',
        'N': 'Unknown',
        'PhD student': 'Unknown',
        'R': 'Unknown',
        'RBRC': 'USA',
        'Roma Sapienza University': 'Italy',
        'Rutgers University': 'USA',
        'SINAP&BNL': 'China',
        'SINAP, BNL': 'China',
        'SINAP/BNL': 'China',
        'SPhT, Saclay': 'France',
        'Shandong University': 'China',
        'Shandong University at Weihai': 'China',
        'Shanghai INstitute of Applied Physics (SINAP), CAS': 'China',
        'Shanghai Institute of Applied Physics': 'China',
        'Shanghai Institute of Applied Physics (SINAP)': 'China',
        'Shanghai Institute of Applied Physics, Chinese Academy of Sciences, P.O. Box 800-204, Shanghai 201800, China': 'China',
        'Sophia Univ': 'Japan',
        'Sophia Univ.': 'Japan',
        'Sophia University, Japan': 'Japan',
        'Sophia university': 'Japan',
        'South China Normal Univeristy': 'China',
        'South China Normal University': 'China',
        'St. Petersburg State Technical University': 'Russia',
        'Stanford University': 'USA',
        'Suranaree University of Technology': 'Thailand',
        'Swansea University': 'UK',
        'TAMU': 'USA',
        'TIFR': 'India',
        'TIFR, Mumbai': 'India',
        'TU Darmstadt': 'Germany',
        'TU Darmstadt / GSI': 'Germany',
        'TU Wien': 'Austria',
        'Tata Institute of Fundamental Research': 'India',
        'Tata Institute, Mumbai, India': 'India',
        'Technion': 'Israel',
        'Technische Universitaet Muenchen': 'Germany',
        'Technische Universität Darmstadt': 'Germany',
        'Texas A&M University + RBRC': 'USA',
        'The University of Tokyo': 'Japan',
        'U': 'Unknown',
        'UCLA': 'USA',
        'UIC': 'USA',
        'USP': 'Brazil',
        'USTC': 'China',
        'USTC && BNL': 'China',
        'USTC, China': 'China',
        'USTC/BNL': 'China',
        'UTFSM': 'Chile',
        'Univ. Tsukuba': 'Japan',
        'Univ. del Piemonte Orientale, Dip.Scienze eTecnologie': 'Italy',
        'Universidad de Santiago de Compostela': 'Spain',
        'Universidade Federal Fluminense': 'Brazil',
        'Universidade Federal Fluminense (UFF), Rio de Janeiro, Brazil': 'Brazil',
        'Universidade Federal do Rio de Janeiro': 'Brazil',
        'Universidade de São Paulo': 'Brazil',
        'Universitat de Barcelona': 'Spain',
        'Universiteit Utrecht': 'Netherlands',
        'University of Bergen': 'Norway',
        'University of Bergen, Norway': 'Norway',
        'University of California (UCD)': 'USA',
        'University of Cape Town': 'South Africa',
        'University of Catania': 'Italy',
        'University of Catania (Italy)': 'Italy',
        'University of Colorado at Boulder': 'USA',
        'University of Houston': 'USA',
        'University of Illinois Chicago': 'USA',
        'University of Illinois Urbana-Champaign': 'USA',
        'University of Jyväskylä': 'Finland',
        'University of Kansas': 'USA',
        'University of Mississippi, Texas A&M University': 'USA',
        'University of North Carolina, Chapel Hill': 'USA',
        'University of Oxford': 'UK',
        'University of Pisa': 'Italy',
        'University of Science and Technology of China': 'China',
        'University of Science and Technology of China (USTC)': 'China',
        'University of Stavanger': 'Norway',
        'University of São Paulo': 'Brazil',
        'University of Tokyo': 'Japan',
        'University of Virginia': 'USA',
        'University of Wrocław': 'Poland',
        'Università La Sapienza di Roma': 'Italy',
        'Università degli Studi di Catania': 'Italy',
        'Universität Bern': 'Switzerland',
        'Université Paris-Saclay': 'France',
        'Universiyu of Oslo': 'Norway',
        'Unknown': 'Unknown',
        'Unknown-Unknown-Unknown': 'Unknown',
        'V': 'Unknown',
        'W': 'Unknown',
        'Warsaw University of Technology': 'Poland',
        'Warsaw Universty of Technology': 'Poland',
        'Weizmann Institute of Science': 'Israel',
        'West University of Timisoara': 'Romania',
        'Westfälische Wilhelms-Universität Münster': 'Germany',
        'Xinyang Normal University': 'China',
        'Yonsei University': 'South Korea',
        'iit-bombay': 'India',
        'l': 'Unknown',
        'lapp-Laboratoire d\'Annecy-le-Vieux de Physique des Particules-In': 'France'
    }
    
    # Load institute-country database
    institute_country_db, normalized_db = load_institute_country_database()
    
    # Add direct mappings to the database
    for institute, country in direct_mappings.items():
        institute_country_db[institute] = country
    
    # Process each conference to collect country and institute data
    years = []
    country_data_plenary = {}
    country_data_parallel = {}
    institute_data_plenary = {}
    institute_data_parallel = {}
    unknown_institutes = set()
    
    for year, data in sorted(all_conference_data.items()):
        years.append(year)
        
        # Process plenary talks
        plenary_talks = data.get('plenary_talks', [])
        country_counts_plenary = {}
        institute_counts_plenary = {}
        
        for talk in plenary_talks:
            institute = talk.get('Institute', '')
            country = talk.get('Country', '')
            
            # Try to determine country from institute if country is missing or unknown
            if (not country or country == 'Unknown') and institute:
                # Check direct mappings first
                if institute in direct_mappings:
                    country = direct_mappings[institute]
                # Then try database
                elif institute in institute_country_db:
                    country = institute_country_db[institute]
                else:
                    # Try case-insensitive match
                    institute_lower = institute.lower()
                    for db_institute, db_country in institute_country_db.items():
                        if db_institute.lower() == institute_lower:
                            country = db_country
                            break
                    
                    # If still not found, try normalized matching
                    if not country or country == 'Unknown':
                        normalized_institute = normalize_institute_name(institute)
                        if normalized_institute in normalized_db:
                            country = normalized_db[normalized_institute]
                        else:
                            # Try partial matching with normalized names
                            for norm_inst, norm_country in normalized_db.items():
                                if (len(norm_inst) > 3 and norm_inst in normalized_institute) or \
                                   (len(normalized_institute) > 3 and normalized_institute in norm_inst):
                                    country = norm_country
                                    break
                
                # Extract country code from parentheses if present
                if (not country or country == 'Unknown') and '(' in institute and ')' in institute:
                    country_code_match = re.search(r'\(([A-Z]{2})\)', institute)
                    if country_code_match:
                        code = country_code_match.group(1)
                        country_map = {
                            'US': 'USA', 'UK': 'United Kingdom', 'DE': 'Germany', 'FR': 'France',
                            'IT': 'Italy', 'JP': 'Japan', 'CN': 'China', 'RU': 'Russia',
                            'CH': 'Switzerland', 'IN': 'India', 'BR': 'Brazil', 'CA': 'Canada',
                            'AU': 'Australia', 'NL': 'Netherlands', 'ES': 'Spain', 'SE': 'Sweden',
                            'DK': 'Denmark', 'NO': 'Norway', 'FI': 'Finland', 'PL': 'Poland',
                            'CZ': 'Czech Republic', 'AT': 'Austria', 'BE': 'Belgium', 'PT': 'Portugal',
                            'GR': 'Greece', 'IL': 'Israel', 'KR': 'South Korea', 'TW': 'Taiwan',
                            'SG': 'Singapore', 'MY': 'Malaysia', 'TH': 'Thailand', 'ZA': 'South Africa',
                            'MX': 'Mexico', 'AR': 'Argentina', 'CL': 'Chile', 'CO': 'Colombia',
                            'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria', 'SK': 'Slovakia',
                            'SI': 'Slovenia', 'HR': 'Croatia', 'RS': 'Serbia', 'UA': 'Ukraine',
                            'TR': 'Turkey', 'IE': 'Ireland', 'IS': 'Iceland', 'LU': 'Luxembourg',
                            'NZ': 'New Zealand'
                        }
                        country = country_map.get(code, code)
                
                if not country or country == 'Unknown':
                    unknown_institutes.add(institute)
            
            # Use 'Unknown' as a last resort
            country = country if country and country != 'Unknown' else 'Unknown'
            institute = institute if institute else 'Unknown'
            
            country_counts_plenary[country] = country_counts_plenary.get(country, 0) + 1
            institute_counts_plenary[institute] = institute_counts_plenary.get(institute, 0) + 1
        
        # Store country and institute data for plenary talks
        country_data_plenary[year] = country_counts_plenary
        institute_data_plenary[year] = institute_counts_plenary
        
        # Process parallel talks (similar logic)
        parallel_talks = data.get('parallel_talks', [])
        country_counts_parallel = {}
        institute_counts_parallel = {}
        
        for talk in parallel_talks:
            institute = talk.get('Institute', '')
            country = talk.get('Country', '')
            
            # Try to determine country from institute if country is missing or unknown
            if (not country or country == 'Unknown') and institute:
                # Check direct mappings first
                if institute in direct_mappings:
                    country = direct_mappings[institute]
                # Then try database
                elif institute in institute_country_db:
                    country = institute_country_db[institute]
                else:
                    # Try case-insensitive match
                    institute_lower = institute.lower()
                    for db_institute, db_country in institute_country_db.items():
                        if db_institute.lower() == institute_lower:
                            country = db_country
                            break
                    
                    # If still not found, try normalized matching
                    if not country or country == 'Unknown':
                        normalized_institute = normalize_institute_name(institute)
                        if normalized_institute in normalized_db:
                            country = normalized_db[normalized_institute]
                        else:
                            # Try partial matching with normalized names
                            for norm_inst, norm_country in normalized_db.items():
                                if (len(norm_inst) > 3 and norm_inst in normalized_institute) or \
                                   (len(normalized_institute) > 3 and normalized_institute in norm_inst):
                                    country = norm_country
                                    break
                
                # Extract country code from parentheses if present
                if (not country or country == 'Unknown') and '(' in institute and ')' in institute:
                    country_code_match = re.search(r'\(([A-Z]{2})\)', institute)
                    if country_code_match:
                        code = country_code_match.group(1)
                        country_map = {
                            'US': 'USA', 'UK': 'United Kingdom', 'DE': 'Germany', 'FR': 'France',
                            'IT': 'Italy', 'JP': 'Japan', 'CN': 'China', 'RU': 'Russia',
                            'CH': 'Switzerland', 'IN': 'India', 'BR': 'Brazil', 'CA': 'Canada',
                            'AU': 'Australia', 'NL': 'Netherlands', 'ES': 'Spain', 'SE': 'Sweden',
                            'DK': 'Denmark', 'NO': 'Norway', 'FI': 'Finland', 'PL': 'Poland',
                            'CZ': 'Czech Republic', 'AT': 'Austria', 'BE': 'Belgium', 'PT': 'Portugal',
                            'GR': 'Greece', 'IL': 'Israel', 'KR': 'South Korea', 'TW': 'Taiwan',
                            'SG': 'Singapore', 'MY': 'Malaysia', 'TH': 'Thailand', 'ZA': 'South Africa',
                            'MX': 'Mexico', 'AR': 'Argentina', 'CL': 'Chile', 'CO': 'Colombia',
                            'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria', 'SK': 'Slovakia',
                            'SI': 'Slovenia', 'HR': 'Croatia', 'RS': 'Serbia', 'UA': 'Ukraine',
                            'TR': 'Turkey', 'IE': 'Ireland', 'IS': 'Iceland', 'LU': 'Luxembourg',
                            'NZ': 'New Zealand'
                        }
                        country = country_map.get(code, code)
                
                if not country or country == 'Unknown':
                    unknown_institutes.add(institute)
            
            # Use 'Unknown' as a last resort
            country = country if country and country != 'Unknown' else 'Unknown'
            institute = institute if institute else 'Unknown'
            
            country_counts_parallel[country] = country_counts_parallel.get(country, 0) + 1
            institute_counts_parallel[institute] = institute_counts_parallel.get(institute, 0) + 1
        
        # Store country and institute data for parallel talks
        country_data_parallel[year] = country_counts_parallel
        institute_data_parallel[year] = institute_counts_parallel
    
    # Update database with unknown institutes
    if unknown_institutes:
        print(f"\nFound {len(unknown_institutes)} institutes without country mapping:")
        for inst in sorted(unknown_institutes):
            print(f"  - '{inst}'")
        
        # Optionally, write to a file for easier processing
        with open('remaining_unknown_institutes.txt', 'w', encoding='utf-8') as f:
            f.write("# Remaining institutes without country mapping\n")
            f.write("# Format: Institute,Country\n")
            for inst in sorted(unknown_institutes):
                f.write(f"{inst},\n")
        print(f"\nList of unknown institutes saved to 'remaining_unknown_institutes.txt'")
    
    # Create plots for country distribution
    create_country_plot(years, country_data_plenary, country_data_parallel, figures_dir)
    
    # Create plots for institute distribution
    create_institute_plot(years, institute_data_plenary, institute_data_parallel, figures_dir)

def create_country_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plots showing distribution of talks by country"""
    # Get top countries across all years
    all_countries = set()
    country_total_counts = {}
    
    for year in years:
        plenary_counts = plenary_data.get(year, {})
        parallel_counts = parallel_data.get(year, {})
        
        for country, count in plenary_counts.items():
            all_countries.add(country)
            country_total_counts[country] = country_total_counts.get(country, 0) + count
            
        for country, count in parallel_counts.items():
            all_countries.add(country)
            country_total_counts[country] = country_total_counts.get(country, 0) + count
    
    # Get top 10 countries by total count
    top_countries = [country for country, _ in sorted(country_total_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # Create plot for plenary talks by country
    plt.figure(figsize=(15, 10))
    
    # Prepare data for plotting
    country_year_data = {country: [] for country in top_countries}
    
    for year in years:
        plenary_counts = plenary_data.get(year, {})
        for country in top_countries:
            country_year_data[country].append(plenary_counts.get(country, 0))
    
    # Create stacked bar chart
    bottom = np.zeros(len(years))
    
    for i, country in enumerate(top_countries):
        plt.bar(years, country_year_data[country], bottom=bottom, label=country)
        bottom += np.array(country_year_data[country])
    
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Plenary Talks')
    plt.title('Distribution of Plenary Talks by Country')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'plenary_talks_by_country.pdf'), format='pdf')
    plt.close()
    
    # Create plot for parallel talks by country
    plt.figure(figsize=(15, 10))
    
    # Prepare data for plotting
    country_year_data = {country: [] for country in top_countries}
    
    for year in years:
        parallel_counts = parallel_data.get(year, {})
        for country in top_countries:
            country_year_data[country].append(parallel_counts.get(country, 0))
    
    # Create stacked bar chart
    bottom = np.zeros(len(years))
    
    for i, country in enumerate(top_countries):
        plt.bar(years, country_year_data[country], bottom=bottom, label=country)
        bottom += np.array(country_year_data[country])
    
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Parallel Talks')
    plt.title('Distribution of Parallel Talks by Country')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'parallel_talks_by_country.pdf'), format='pdf')
    plt.close()

def create_institute_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plots showing distribution of talks by institute"""
    # Get top institutes across all years
    all_institutes = set()
    institute_total_counts = {}
    
    for year in years:
        plenary_counts = plenary_data.get(year, {})
        parallel_counts = parallel_data.get(year, {})
        
        for institute, count in plenary_counts.items():
            all_institutes.add(institute)
            institute_total_counts[institute] = institute_total_counts.get(institute, 0) + count
            
        for institute, count in parallel_counts.items():
            all_institutes.add(institute)
            institute_total_counts[institute] = institute_total_counts.get(institute, 0) + count
    
    # Get top 15 institutes by total count
    top_institutes = [institute for institute, _ in sorted(institute_total_counts.items(), key=lambda x: x[1], reverse=True)[:15]]
    
    # Create plot for plenary talks by institute
    plt.figure(figsize=(18, 12))
    
    # Prepare data for plotting
    institute_year_data = {institute: [] for institute in top_institutes}
    
    for year in years:
        plenary_counts = plenary_data.get(year, {})
        for institute in top_institutes:
            institute_year_data[institute].append(plenary_counts.get(institute, 0))
    
    # Create stacked bar chart
    bottom = np.zeros(len(years))
    
    for i, institute in enumerate(top_institutes):
        plt.bar(years, institute_year_data[institute], bottom=bottom, label=institute)
        bottom += np.array(institute_year_data[institute])
    
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Plenary Talks')
    plt.title('Distribution of Plenary Talks by Institute')
    plt.legend(title='Institute', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'plenary_talks_by_institute.pdf'), format='pdf')
    plt.close()
    
    # Create plot for parallel talks by institute
    plt.figure(figsize=(18, 12))
    
    # Prepare data for plotting
    institute_year_data = {institute: [] for institute in top_institutes}
    
    for year in years:
        parallel_counts = parallel_data.get(year, {})
        for institute in top_institutes:
            institute_year_data[institute].append(parallel_counts.get(institute, 0))
    
    # Create stacked bar chart
    bottom = np.zeros(len(years))
    
    for i, institute in enumerate(top_institutes):
        plt.bar(years, institute_year_data[institute], bottom=bottom, label=institute)
        bottom += np.array(institute_year_data[institute])
    
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Parallel Talks')
    plt.title('Distribution of Parallel Talks by Institute')
    plt.legend(title='Institute', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'parallel_talks_by_institute.pdf'), format='pdf')
    plt.close()

def create_venue_plot(all_conference_data):
    """Create a plot showing conference venues over time"""
    print("\nCreating conference venue plot...")
    
    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Gather venue data
    years = []
    venues = []
    countries = []
    continents = {
        'USA': 'North America',
        'China': 'Asia',
        'Italy': 'Europe',
        'Germany': 'Europe',
        'Japan': 'Asia',
        'Netherlands': 'Europe',
        'France': 'Europe',
        'Poland': 'Europe',
        'Denmark': 'Europe',
        'Canada': 'North America',
        'India': 'Asia',
        'South Korea': 'Asia',
        'Australia': 'Oceania'
    }
    continent_colors = {
        'North America': '#1f77b4',  # blue
        'Europe': '#ff7f0e',        # orange
        'Asia': '#2ca02c',          # green
        'Oceania': '#d62728',       # red
        'Other': '#9467bd'          # purple
    }
    
    for year, data in sorted(all_conference_data.items()):
        years.append(year)
        location = data.get('location', '')
        venues.append(location)
        
        # Extract country from location (assuming format "City, Country")
        if ',' in location:
            country = location.split(',')[-1].strip()
        else:
            country = location  # Use full location if no comma
        countries.append(country)
    
    # Create the venue plot
    plt.figure(figsize=(12, 6))
    
    # Plot points
    for i, (year, venue, country) in enumerate(zip(years, venues, countries)):
        continent = continents.get(country, 'Other')
        color = continent_colors.get(continent, '#9467bd')
        plt.scatter(i, 0, s=200, color=color, edgecolor='black', zorder=3)
        
        # Add venue labels
        plt.annotate(venue, xy=(i, 0), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        # Add year labels
        plt.annotate(f"QM{year}", xy=(i, 0), xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=10)
    
    # Create a custom legend for continents
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 label=continent)
                      for continent, color in continent_colors.items()]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=5, title="Continents")
    
    # Remove axes but keep the frame
    plt.yticks([])
    plt.xticks([])
    plt.ylim(-1, 1)
    plt.xlim(-0.5, len(years) - 0.5)
    
    # Add a horizontal line for timeline
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7, zorder=1)
    
    # Add title
    plt.title('Quark Matter Conference Venues (2011-2025)', fontsize=14, pad=30)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'conference_venues.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Conference venue plot saved to {save_path}")

def fetch_and_analyze_conferences():
    print("=== FETCHING AND PROCESSING CONFERENCES ===")
    try:
        with open('listofQMindigo', 'r') as f:
            conferences = [line.strip().split()[:2] for line in f if not line.strip().startswith('#')]
            
        conference_data = {}
        processed_conferences = []
        
        for year, indico_id in conferences:
            print(f"\nProcessing QM{year} (ID: {indico_id})")
            processed_file = f'data/QM{year}_processed_data.json'
            
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conference_data[year] = {
                        'all_talks': data['all_talks'],
                        'plenary_talks': data['plenary_talks'],
                        'parallel_talks': data['parallel_talks'],
                        'poster_talks': data['poster_talks'],
                        'flash_talks': data['flash_talks'],
                        'metadata': data['metadata']
                    }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  Could not load processed data: {e}")
                continue
        
        # Print final summary table
        print("\nYear Location Total Plenary Parallel Poster Flash")
        print("-" * 65)
        
        # Sort conferences by year
        conferences.sort(key=lambda x: x[0])
        
        for year, _ in conferences:
            if year in conference_data:
                data = conference_data[year]
                location = CONFERENCE_LOCATIONS.get(year, 'Unknown location')
                total = len(data['all_talks'])
                plenary = len(data['plenary_talks'])
                parallel = len(data['parallel_talks'])
                poster = len(data['poster_talks'])
                flash = len(data['flash_talks'])
                print(f"{year} {location} {total} {plenary} {parallel} {poster} {flash}")
        
        print("\nUnknown Institutes Analysis:")
        print("-" * 65)
        for year, _ in conferences:
            if year in conference_data:
                print(f"\nQM{year}:")
                # Get unknown institutes for plenary talks
                unknown_plenary = [talk['Institute'] for talk in conference_data[year]['plenary_talks'] 
                                 if talk['Country'] == 'Unknown']
                if unknown_plenary:
                    print(f"  Plenary unknown institutes ({len(unknown_plenary)}):")
                    for inst in sorted(set(unknown_plenary))[:5]:
                        print(f"    - {inst}")
                    if len(unknown_plenary) > 5:
                        print(f"    ... and {len(set(unknown_plenary)) - 5} more")
                
                # Get unknown institutes for parallel talks
                unknown_parallel = [talk['Institute'] for talk in conference_data[year]['parallel_talks'] 
                                 if talk['Country'] == 'Unknown']
                if unknown_parallel:
                    print(f"  Parallel unknown institutes ({len(unknown_parallel)}):")
                    for inst in sorted(set(unknown_parallel))[:5]:
                        print(f"    - {inst}")
                    if len(unknown_parallel) > 5:
                        print(f"    ... and {len(set(unknown_parallel)) - 5} more")
                
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1)

if __name__ == "__main__":
    try:
        with open('listofQMindigo', 'r') as f:
            conferences = [line.strip().split()[:2] for line in f if not line.strip().startswith('#')]
            
        conference_data = {}
        processed_conferences = []
        
        for year, indico_id in conferences:
            conference_stats = fetch_and_process_contributions(indico_id, year)
            if conference_stats:
                conference_data[year] = conference_stats
        
        # Print final summary table
        print("\nYear Location Total Plenary Parallel Poster Flash Unk_Plen Unk_Par")
        print("-" * 85)
        
        # Sort conferences by year
        conferences.sort(key=lambda x: x[0])
        
        for year, _ in conferences:
            if year in conference_data:
                data = conference_data[year]
                location = CONFERENCE_LOCATIONS.get(year, 'Unknown location')
                total = len(data['all_talks'])
                plenary = len(data['plenary_talks'])
                parallel = len(data['parallel_talks'])
                poster = len(data['poster_talks'])
                flash = FLASH_TALK_COUNTS.get(year, 0)  # Get flash talk count from manual dictionary
                
                # Count unknown institutes
                unknown_plenary = len([t for t in data['plenary_talks'] if t['Institute'] == 'Unknown'])
                unknown_parallel = len([t for t in data['parallel_talks'] if t['Institute'] == 'Unknown'])
                
                print(f"{year} {location:<25} {total:<6} {plenary:<8} {parallel:<8} {poster:<6} {flash:<5} {unknown_plenary:<8} {unknown_parallel}")
        
        # Generate plot
        plot_conference_statistics(conference_data)
        print_summary(conferences)
        
        # Create the keywords plot
        create_keywords_plot(conference_data)
        
        # Create country and institute plots
        create_country_institute_plots(conference_data)
        
        # Create venue plot
        create_venue_plot(conference_data)
                
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 