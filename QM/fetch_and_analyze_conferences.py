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
import pandas as pd

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


def plot_talks_by_country(contributions, title, filename, top_n=15, show_others=True, 
                          color_palette="viridis", by_region=False, include_percentages=True):
    """
    Enhanced plot of talks by country with more countries, improved styling, and optional grouping by region.
    
    Parameters:
    - contributions: DataFrame containing contribution data
    - title: Plot title
    - filename: Output filename
    - top_n: Number of top countries to show individually (increased from 10 to 15)
    - show_others: Whether to aggregate remaining countries as "Others"
    - color_palette: Matplotlib colormap name for the bars
    - by_region: If True, organize countries by geographical region
    - include_percentages: If True, add percentage labels to bars
    """
    # Count contributions by country
    country_counts = contributions['country'].value_counts()
    total_talks = len(contributions)
    
    # Define regions for organizing countries if needed
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'UK', 'France', 'Italy', 'Poland', 'Switzerland', 'Netherlands', 
                  'Czech Republic', 'Spain', 'Belgium', 'Sweden', 'Finland', 'Denmark', 'Norway', 
                  'Austria', 'Hungary', 'Romania', 'Portugal', 'Greece', 'Ireland'],
        'Asia': ['China', 'Japan', 'Korea', 'India', 'Israel', 'Turkey', 'Taiwan', 'Singapore', 
                'Vietnam', 'Thailand', 'Malaysia', 'Indonesia'],
        'Other': ['Brazil', 'Australia', 'South Africa', 'Argentina', 'Chile', 'Egypt', 
                 'New Zealand', 'Russia', 'Ukraine', 'Belarus']
    }
    
    # Setup the plot with larger figure size
    plt.figure(figsize=(12, 8))
    
    if by_region:
        # Group countries by region
        region_data = {}
        for region, countries in regions.items():
            region_data[region] = country_counts[country_counts.index.isin(countries)].sum()
        
        # Create a stacked bar chart by region
        # Implementation details would go here
        pass
    else:
        # Get top N countries and calculate percentages
        top_countries = country_counts.head(top_n)
        top_percentages = (top_countries / total_talks * 100).round(1)
        
        # If showing others, calculate their aggregate
        if show_others and len(country_counts) > top_n:
            others_count = country_counts.iloc[top_n:].sum()
            top_countries = top_countries.append(pd.Series([others_count], index=['Others']))
            top_percentages = top_percentages.append(pd.Series([others_count / total_talks * 100], index=['Others']))
        
        # Create color map - using a perceptually uniform colormap
        cmap = plt.cm.get_cmap(color_palette, len(top_countries))
        colors = [cmap(i) for i in range(len(top_countries))]
        
        # Create horizontal bar chart for better readability
        bars = plt.barh(top_countries.index[::-1], top_countries.values[::-1], color=colors[::-1])
        
        # Add percentage labels if requested
        if include_percentages:
            for i, (bar, percentage) in enumerate(zip(bars, top_percentages.values[::-1])):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                         f'{percentage:.1f}%', va='center', fontsize=9)
    
    # Enhanced styling
    plt.xlabel('Number of Talks', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add regional color coding in legend if not grouped by region
    if not by_region:
        from matplotlib.patches import Patch
        legend_elements = []
        for region, countries in regions.items():
            countries_in_plot = [c for c in countries if c in top_countries.index]
            if countries_in_plot:
                legend_elements.append(Patch(facecolor=cmap(0.5), edgecolor='black', 
                                            label=f'{region} ({len(countries_in_plot)} countries)'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='lower right', title='Regions')
    
    # Ensure we save to figures directory
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, os.path.basename(filename))
    
    # Save the figure to figures directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {save_path}")

def plot_talks_by_institute(contributions, title, filename, top_n=20, show_others=True,
                           color_palette="viridis", include_percentages=True):
    """
    Enhanced plot of talks by institute with more institutes and improved styling.
    
    Parameters:
    - contributions: DataFrame containing contribution data
    - title: Plot title
    - filename: Output filename
    - top_n: Number of top institutes to show individually (increased from 10 to 20)
    - show_others: Whether to aggregate remaining institutes as "Others"
    - color_palette: Matplotlib colormap name for the bars
    - include_percentages: If True, add percentage labels to bars
    """
    # Count contributions by institute
    institute_counts = contributions['institute'].value_counts()
    total_talks = len(contributions)
    
    # Get top N institutes and calculate percentages
    top_institutes = institute_counts.head(top_n)
    top_percentages = (top_institutes / total_talks * 100).round(1)
    
    # If showing others, calculate their aggregate
    if show_others and len(institute_counts) > top_n:
        others_count = institute_counts.iloc[top_n:].sum()
        top_institutes = top_institutes.append(pd.Series([others_count], index=['Others']))
        top_percentages = top_percentages.append(pd.Series([others_count / total_talks * 100], index=['Others']))
    
    # Setup the plot with larger figure size for more institutes
    plt.figure(figsize=(14, 10))
    
    # Create color map - using a perceptually uniform colormap
    cmap = plt.cm.get_cmap(color_palette, len(top_institutes))
    colors = [cmap(i) for i in range(len(top_institutes))]
    
    # Create horizontal bar chart for better readability of institute names
    bars = plt.barh(top_institutes.index[::-1], top_institutes.values[::-1], color=colors[::-1])
    
    # Add percentage labels
    for i, (bar, percentage) in enumerate(zip(bars, top_percentages.values[::-1])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{percentage:.1f}%', va='center', fontsize=9)
    
    # Categorize institutes by type and show in the legend
    institute_types = {
        'National Laboratory': ['BNL', 'CERN', 'LBNL', 'ORNL', 'LANL', 'ANL', 'FNAL', 'JLAB', 'JINR'],
        'University': ['University', 'College', 'School', 'Institut', 'Universidad'],
        'Research Center': ['Institute', 'Center', 'Centre', 'Laboratory']
    }
    
    # Create custom legend for institute types
    from matplotlib.patches import Patch
    legend_elements = []
    for inst_type, keywords in institute_types.items():
        matching_institutes = [i for i in top_institutes.index 
                              if any(kw in i for kw in keywords) and i != 'Others']
        if matching_institutes:
            legend_elements.append(Patch(facecolor=cmap(0.5), edgecolor='black', 
                                        label=f'{inst_type} ({len(matching_institutes)})'))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right', title='Institute Types')
    
    # Enhanced styling
    plt.xlabel('Number of Talks', fontsize=12)
    plt.ylabel('Institute', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Ensure we save to figures directory
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, os.path.basename(filename))
    
    # Save the figure to figures directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {save_path}")

def create_country_trend_plot(all_contributions_by_year, filename):
    """
    Create a line plot showing trends in country representation over time.
    
    Parameters:
    - all_contributions: Dictionary with year keys and contribution DataFrames as values
    - filename: Output filename
    """
    plt.figure(figsize=(14, 8))
    
    # Get all years and top countries across all conferences
    years = sorted(all_contributions_by_year.keys())
    all_country_counts = {}
    
    for year, contributions in all_contributions_by_year.items():
        country_counts = contributions['country'].value_counts()
        all_country_counts[year] = country_counts
    
    # Identify top 8 countries across all years
    top_countries = pd.concat(all_country_counts.values()).groupby(level=0).sum().nlargest(8).index
    
    # Plot trend for each top country
    cmap = plt.cm.get_cmap('tab10', len(top_countries))
    for i, country in enumerate(top_countries):
        percentages = []
        for year in years:
            counts = all_country_counts[year]
            total = len(all_contributions_by_year[year])
            percentage = counts.get(country, 0) / total * 100 if total > 0 else 0
            percentages.append(percentage)
        
        plt.plot(years, percentages, 'o-', label=country, color=cmap(i), linewidth=2, markersize=8)
    
    # Add emerging countries with interesting trends (showing growth)
    emerging_countries = ['Brazil', 'South Africa', 'Poland', 'Czech Republic']
    emerging_cmap = plt.cm.get_cmap('Paired', len(emerging_countries))
    
    for i, country in enumerate(emerging_countries):
        if country not in top_countries:  # Only add if not already in top countries
            percentages = []
            for year in years:
                counts = all_country_counts[year]
                total = len(all_contributions_by_year[year])
                percentage = counts.get(country, 0) / total * 100 if total > 0 else 0
                percentages.append(percentage)
            
            plt.plot(years, percentages, 's--', label=f"{country} (emerging)", 
                     color=emerging_cmap(i), linewidth=1.5, markersize=6)
    
    # Enhanced styling
    plt.xlabel('Conference Year', fontsize=12)
    plt.ylabel('Percentage of Total Talks (%)', fontsize=12)
    plt.title('Trends in Country Representation at Quark Matter Conferences', 
              fontsize=14, fontweight='bold')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Ensure we save to figures directory
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, os.path.basename(filename))
    
    # Save the figure to figures directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved country trends plot to {save_path}")

def create_institute_bubble_chart(all_contributions_by_year, filename):
    """
    Create a bubble chart showing institute participation across conferences.
    
    Parameters:
    - all_contributions_by_year: Dictionary with years as keys and lists of contributions as values
    - filename: Output filename for the chart
    """
    print("\nCreating institute bubble chart...")
    
    # Ensure figures directory exists
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, os.path.basename(filename))
    
    try:
        # Extract years and sort them
        years = sorted(all_contributions_by_year.keys())
        
        # Count institute appearances by year
        institute_by_year = {}
        for year in years:
            contributions = all_contributions_by_year[year]
            # Create a pandas Series with institute counts
            if isinstance(contributions, pd.DataFrame):
                institute_counts = contributions['institute'].value_counts()
            else:
                # If it's a list of dictionaries, convert to DataFrame first
                contributions_df = pd.DataFrame(contributions)
                institute_counts = contributions_df['institute'].value_counts() if 'institute' in contributions_df.columns else pd.Series()
            
            institute_by_year[year] = institute_counts
        
        # Find institutes that appear frequently
        all_institutes = set()
        for year in years:
            all_institutes.update(institute_by_year[year].index)
        
        # Count total contributions by institute
        institute_totals = {}
        for institute in all_institutes:
            total = sum(institute_by_year[year].get(institute, 0) for year in years)
            institute_totals[institute] = total
        
        # Select top 30 institutes by total contributions
        top_institutes = sorted(institute_totals.items(), key=lambda x: x[1], reverse=True)[:30]
        top_institute_names = [i[0] for i in top_institutes]
        
        print(f"Selected top {len(top_institute_names)} institutes for bubble chart")
        
        # Create a bubble chart
        plt.figure(figsize=(15, 10))
        
        # Set up a grid for the institutes (rows) and years (columns)
        num_institutes = len(top_institute_names)
        num_years = len(years)
        
        # Create normalized colormap for bubble sizes
        sizes = []
        for year in years:
            for institute in top_institute_names:
                count = institute_by_year[year].get(institute, 0)
                if count > 0:
                    sizes.append(count)
        
        max_size = max(sizes) if sizes else 1
        print(f"Maximum contributions from a single institute in one year: {max_size}")
        
        # Plot the bubbles
        for i, institute in enumerate(top_institute_names):
            for j, year in enumerate(years):
                count = institute_by_year[year].get(institute, 0)
                if count > 0:
                    size = (count / max_size) * 300  # Scale bubble size
                    plt.scatter(j, num_institutes - i - 1, s=size, alpha=0.7, 
                               edgecolor='black', linewidth=0.5)
                    if count > max_size * 0.3:  # Only show count for larger bubbles
                        plt.text(j, num_institutes - i - 1, str(count), 
                                 ha='center', va='center', fontsize=8)
        
        # Enhanced styling
        plt.xticks(range(num_years), years, rotation=45)
        plt.yticks(range(num_institutes), top_institute_names[::-1])
        plt.grid(linestyle='--', alpha=0.3)
        plt.title('Institute Contributions Across Quark Matter Conferences',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Conference Year', fontsize=12)
        plt.tight_layout()
        
        # Add bubble size legend
        for size in [1, 5, 10, 20]:
            if size <= max_size:
                scaled_size = (size / max_size) * 300
                plt.scatter([], [], s=scaled_size, alpha=0.7, edgecolor='black', 
                           linewidth=0.5, label=f'{size} talks')
        plt.legend(title='Contribution Count', loc='upper right', scatterpoints=1)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Institute bubble chart saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating institute bubble chart: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating institute bubble chart:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

def create_country_trends_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plots showing country representation trends over time"""
    print("\nCreating country trends plot...")
    save_path = os.path.join(figures_dir, "country_trends_over_time.pdf")
    
    try:
        # Combine plenary and parallel data for total counts
        all_contributions_by_year = {}
        country_data = {}
        
        for year in years:
            # Get plenary and parallel counts for this year
            plenary_counts = plenary_data.get(year, {})
            parallel_counts = parallel_data.get(year, {})
            
            # Count total talks for this year
            total_talks = sum(plenary_counts.values()) + sum(parallel_counts.values())
            
            # Combine counts from plenary and parallel
            combined_counts = {}
            for country, count in plenary_counts.items():
                combined_counts[country] = combined_counts.get(country, 0) + count
                
            for country, count in parallel_counts.items():
                combined_counts[country] = combined_counts.get(country, 0) + count
            
            # Calculate percentages
            country_percentages = {}
            for country, count in combined_counts.items():
                percentage = (count / total_talks * 100) if total_talks > 0 else 0
                country_percentages[country] = round(percentage, 1)
            
            country_data[year] = country_percentages
            
            # Store talks for potential future use
            all_contributions_by_year[year] = total_talks
            
            print(f"  Year {year}: Found {len(combined_counts)} unique countries")
        
        # Find all unique countries across all years
        all_countries = set()
        for year in years:
            all_countries.update(country_data[year].keys())
        
        print(f"Found {len(all_countries)} unique countries across all years")
        
        # Count total contributions by country
        country_totals = {}
        for country in all_countries:
            total = sum(country_data[year].get(country, 0) * all_contributions_by_year[year] / 100 
                      for year in years)
            country_totals[country] = total
        
        # Select top 8 countries by total contributions for main lines
        top_countries = sorted(country_totals.items(), key=lambda x: x[1], reverse=True)[:8]
        top_country_names = [c[0] for c in top_countries]
        
        # Select emerging countries for dashed lines (countries ranked 9-15)
        emerging_countries = sorted(country_totals.items(), key=lambda x: x[1], reverse=True)[8:15]
        emerging_country_names = [c[0] for c in emerging_countries]
        
        print(f"Selected top {len(top_country_names)} countries for main trend lines")
        print(f"Selected {len(emerging_country_names)} emerging countries for dashed trend lines")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create colormap for top countries
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_country_names)))
        
        # Plot the trends for top countries
        for i, country in enumerate(top_country_names):
            country_percentages = [country_data[year].get(country, 0) for year in years]
            plt.plot(years, country_percentages, marker='o', linewidth=2, color=colors[i], label=country)
        
        # Plot the trends for emerging countries with dashed lines
        colors_emerging = plt.cm.Set2(np.linspace(0, 1, len(emerging_country_names)))
        for i, country in enumerate(emerging_country_names):
            country_percentages = [country_data[year].get(country, 0) for year in years]
            plt.plot(years, country_percentages, marker='s', linewidth=1.5, 
                    linestyle='--', color=colors_emerging[i], label=country)
        
        # Add styling
        plt.title('Country Representation Trends Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Conference Year', fontsize=12)
        plt.ylabel('Percentage of Contributions', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust spacing for the legend
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Country trends chart saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating country trends chart: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating country trends chart:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

def create_institute_bubble_plot(years, plenary_data, parallel_data, figures_dir):
    """Create bubble chart showing institute participation across conferences"""
    print("\nCreating institute bubble chart...")
    save_path = os.path.join(figures_dir, "institute_bubble_chart.pdf")
    
    try:
        # Combine plenary and parallel data for total counts
        institute_by_year = {}
        
        for year in years:
            # Get plenary and parallel counts for this year
            plenary_counts = plenary_data.get(year, {})
            parallel_counts = parallel_data.get(year, {})
            
            # Combine counts from plenary and parallel
            combined_counts = {}
            for institute, count in plenary_counts.items():
                combined_counts[institute] = combined_counts.get(institute, 0) + count
                
            for institute, count in parallel_counts.items():
                combined_counts[institute] = combined_counts.get(institute, 0) + count
            
            institute_by_year[year] = combined_counts
            print(f"  Year {year}: Found {len(combined_counts)} unique institutes")
        
        # Find all unique institutes across all years
        all_institutes = set()
        for year in years:
            all_institutes.update(institute_by_year[year].keys())
        
        print(f"Found {len(all_institutes)} unique institutes across all years")
        
        # Count total contributions by institute
        institute_totals = {}
        for institute in all_institutes:
            total = sum(institute_by_year[year].get(institute, 0) for year in years)
            institute_totals[institute] = total
        
        # Select top 30 institutes by total contributions
        top_institutes = sorted(institute_totals.items(), key=lambda x: x[1], reverse=True)[:30]
        top_institute_names = [i[0] for i in top_institutes]
        
        print(f"Selected top {len(top_institute_names)} institutes for bubble chart")
        print("Top 5 institutes:")
        for inst, count in top_institutes[:5]:
            print(f"  {inst}: {count} talks")
        
        # Create a bubble chart
        plt.figure(figsize=(15, 10))
        
        # Set up a grid for the institutes (rows) and years (columns)
        num_institutes = len(top_institute_names)
        num_years = len(years)
        
        # Create normalized colormap for bubble sizes
        sizes = []
        for year in years:
            for institute in top_institute_names:
                count = institute_by_year[year].get(institute, 0)
                if count > 0:
                    sizes.append(count)
        
        max_size = max(sizes) if sizes else 1
        print(f"Maximum contributions from a single institute in one year: {max_size}")
        
        # Plot the bubbles
        for i, institute in enumerate(top_institute_names):
            for j, year in enumerate(years):
                count = institute_by_year[year].get(institute, 0)
                if count > 0:
                    size = (count / max_size) * 300  # Scale bubble size
                    plt.scatter(j, num_institutes - i - 1, s=size, alpha=0.7, 
                               edgecolor='black', linewidth=0.5)
                    if count > max_size * 0.3:  # Only show count for larger bubbles
                        plt.text(j, num_institutes - i - 1, str(count), 
                                 ha='center', va='center', fontsize=8)
        
        # Enhanced styling
        plt.xticks(range(num_years), years, rotation=45)
        plt.yticks(range(num_institutes), top_institute_names[::-1])
        plt.grid(linestyle='--', alpha=0.3)
        plt.title('Institute Contributions Across Quark Matter Conferences',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Conference Year', fontsize=12)
        plt.tight_layout()
        
        # Add bubble size legend
        for size in [1, 5, 10, 20]:
            if size <= max_size:
                scaled_size = (size / max_size) * 300
                plt.scatter([], [], s=scaled_size, alpha=0.7, edgecolor='black', 
                           linewidth=0.5, label=f'{size} talks')
        plt.legend(title='Contribution Count', loc='upper right', scatterpoints=1)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Institute bubble chart saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating institute bubble chart: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating institute bubble chart:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

def create_regional_diversity_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plot showing regional diversity across conferences"""
    print("\nCreating regional diversity plot...")
    save_path = os.path.join(figures_dir, "regional_diversity.pdf")
    
    try:
        # Define regions and their countries
        regions = {
            'North America': ['USA', 'Canada', 'Mexico'],
            'Europe': ['Germany', 'UK', 'France', 'Italy', 'Spain', 'Switzerland', 'Poland', 
                      'Netherlands', 'Czech Republic', 'Sweden', 'Finland', 'Denmark', 'Norway',
                      'Belgium', 'Austria', 'Hungary', 'Portugal', 'Greece', 'Ireland', 'Romania',
                      'Slovakia', 'Serbia', 'Croatia', 'Ukraine', 'Belarus', 'Bulgaria', 'Slovenia',
                      'Luxembourg', 'Lithuania', 'Estonia', 'Latvia', 'Iceland', 'Russia'],
            'East Asia': ['Japan', 'China', 'South Korea', 'Taiwan', 'Hong Kong'],
            'South & SE Asia': ['India', 'Singapore', 'Malaysia', 'Thailand', 'Vietnam', 'Indonesia',
                              'Philippines', 'Bangladesh', 'Pakistan', 'Nepal', 'Sri Lanka'],
            'Middle East': ['Israel', 'Turkey', 'Iran', 'Qatar', 'UAE', 'Saudi Arabia', 'Lebanon',
                          'Jordan', 'Kuwait', 'Oman', 'Iraq'],
            'Oceania': ['Australia', 'New Zealand'],
            'Latin America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela',
                            'Mexico', 'Cuba', 'Costa Rica', 'Ecuador'],
            'Africa': ['South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Morocco', 'Tunisia',
                      'Algeria', 'Ghana', 'Ethiopia', 'Tanzania']
        }
        
        # Create a dictionary to map countries to regions
        country_to_region = {}
        for region, countries in regions.items():
            for country in countries:
                country_to_region[country] = region
        
        # Initialize data structures for region counts
        region_counts_total = {}
        region_counts_by_year = {year: {} for year in years}
        
        # Process data for all years
        for year in years:
            plenary_counts = plenary_data.get(year, {})
            parallel_counts = parallel_data.get(year, {})
            
            # Combine counts
            combined_counts = {}
            for country, count in plenary_counts.items():
                combined_counts[country] = combined_counts.get(country, 0) + count
            for country, count in parallel_counts.items():
                combined_counts[country] = combined_counts.get(country, 0) + count
            
            # Map countries to regions and count
            for country, count in combined_counts.items():
                if country != 'Unknown':
                    region = country_to_region.get(country, 'Other')
                else:
                    region = 'Unknown'
                
                # Add to total counts
                region_counts_total[region] = region_counts_total.get(region, 0) + count
                
                # Add to yearly counts
                region_counts_by_year[year][region] = region_counts_by_year[year].get(region, 0) + count
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # Add subplots in a 1x2 grid
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # 1. Pie chart for overall regional distribution
        regions_to_plot = {k: v for k, v in region_counts_total.items() if k != 'Unknown' and v > 0}
        labels = list(regions_to_plot.keys())
        sizes = list(regions_to_plot.values())
        
        # Sort by size
        sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_data]
        sizes = [x[1] for x in sorted_data]
        
        # Custom colors for regions
        region_colors = {
            'North America': '#1f77b4',    # blue
            'Europe': '#ff7f0e',           # orange
            'East Asia': '#2ca02c',        # green
            'South & SE Asia': '#d62728',  # red
            'Middle East': '#9467bd',      # purple
            'Oceania': '#8c564b',          # brown
            'Latin America': '#e377c2',    # pink
            'Africa': '#7f7f7f',           # grey
            'Other': '#bcbd22',            # olive
            'Unknown': '#17becf'           # cyan
        }
        
        colors = [region_colors.get(region, '#17becf') for region in labels]
        
        # Create pie chart
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        ax1.set_title('Overall Regional Distribution', fontsize=14, fontweight='bold')
        
        # 2. Stacked area chart for trend over time
        # Convert year-by-year data to a format suitable for stacked area
        region_data = {region: [] for region in labels}
        for year in years:
            year_total = sum(region_counts_by_year[year].values())
            for region in labels:
                count = region_counts_by_year[year].get(region, 0)
                percentage = (count / year_total * 100) if year_total > 0 else 0
                region_data[region].append(percentage)
        
        # Create the stacked area chart
        bottom = np.zeros(len(years))
        for region in labels:
            ax2.fill_between(years, bottom, bottom + np.array(region_data[region]), 
                            label=region, color=region_colors.get(region, '#17becf'), alpha=0.8)
            bottom += np.array(region_data[region])
        
        # Style the area chart
        ax2.set_xlabel('Conference Year', fontsize=12)
        ax2.set_ylabel('Percentage of Contributions', fontsize=12)
        ax2.set_title('Regional Representation Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(title='Regions', loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Regional diversity plot saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating regional diversity plot: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating regional diversity plot:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

def create_diversity_metrics_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plot showing diversity metrics over time"""
    print("\nCreating diversity metrics plot...")
    save_path = os.path.join(figures_dir, "diversity_metrics.pdf")
    
    try:
        # Initialize metrics holders
        unique_countries_by_year = {}
        hhi_by_year = {}  # Herfindahl-Hirschman Index, a measure of concentration
        
        # Process data for all years
        for year in years:
            plenary_counts = plenary_data.get(year, {})
            parallel_counts = parallel_data.get(year, {})
            
            # Combine counts
            country_counts = {}
            for country, count in plenary_counts.items():
                if country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + count
                    
            for country, count in parallel_counts.items():
                if country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + count
            
            # Number of unique countries
            unique_countries_by_year[year] = len(country_counts)
            
            # Calculate HHI (sum of squared market shares)
            total_contributions = sum(country_counts.values())
            hhi = 0
            if total_contributions > 0:
                for count in country_counts.values():
                    market_share = count / total_contributions
                    hhi += market_share ** 2
            hhi_by_year[year] = hhi * 10000  # Scale to 0-10000 range
            
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Number of unique countries
        ax1.plot(years, [unique_countries_by_year.get(year, 0) for year in years], 
                'o-', linewidth=2, color='#1f77b4')
        
        # Add data labels
        for i, year in enumerate(years):
            count = unique_countries_by_year.get(year, 0)
            ax1.annotate(f'{count}', (year, count), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9)
        
        # Style the first plot
        ax1.set_ylabel('Number of Countries', fontsize=12)
        ax1.set_title('Country Diversity Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: HHI index (lower is more diverse)
        ax2.plot(years, [hhi_by_year.get(year, 0) for year in years], 
                'o-', linewidth=2, color='#ff7f0e')
        
        # Add data labels
        for i, year in enumerate(years):
            hhi = hhi_by_year.get(year, 0)
            ax2.annotate(f'{hhi:.0f}', (year, hhi), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9)
        
        # Style the second plot
        ax2.set_xlabel('Conference Year', fontsize=12)
        ax2.set_ylabel('HHI Index', fontsize=12)
        ax2.set_title('Concentration Index Over Time (Lower = More Diverse)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanatory text about HHI
        ax2.text(0.5, -0.15, 
                "The Herfindahl-Hirschman Index (HHI) measures market concentration.\n" +
                "In this context, lower values indicate more diverse country representation.",
                transform=ax2.transAxes, ha='center', va='center', fontsize=9)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diversity metrics plot saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating diversity metrics plot: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating diversity metrics plot:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

def create_representation_ratio_plot(years, plenary_data, parallel_data, figures_dir):
    """Create plot showing representation ratio between plenary and parallel talks"""
    print("\nCreating representation ratio plot...")
    save_path = os.path.join(figures_dir, "representation_ratio.pdf")
    
    try:
        # Aggregate data across all years to get overall counts
        plenary_counts_total = {}
        parallel_counts_total = {}
        
        for year in years:
            # Add plenary counts
            for country, count in plenary_data.get(year, {}).items():
                if country != 'Unknown':
                    plenary_counts_total[country] = plenary_counts_total.get(country, 0) + count
            
            # Add parallel counts
            for country, count in parallel_data.get(year, {}).items():
                if country != 'Unknown':
                    parallel_counts_total[country] = parallel_counts_total.get(country, 0) + count
        
        # Calculate total talks for percentages
        total_plenary = sum(plenary_counts_total.values())
        total_parallel = sum(parallel_counts_total.values())
        
        # Calculate percentages and ratios for each country
        countries = set(plenary_counts_total.keys()) | set(parallel_counts_total.keys())
        representation_data = {}
        
        for country in countries:
            # Get counts (default to 0 if not present)
            plenary_count = plenary_counts_total.get(country, 0)
            parallel_count = parallel_counts_total.get(country, 0)
            
            # Only include countries with at least 5 talks total
            if plenary_count + parallel_count >= 5:
                # Calculate percentages
                plenary_pct = (plenary_count / total_plenary * 100) if total_plenary > 0 else 0
                parallel_pct = (parallel_count / total_parallel * 100) if total_parallel > 0 else 0
                
                # Calculate ratio (avoid division by zero)
                if parallel_pct > 0:
                    ratio = plenary_pct / parallel_pct
                else:
                    ratio = float('inf') if plenary_pct > 0 else 1.0  # Handle division by zero
                
                # Store data for plotting
                representation_data[country] = {
                    'plenary_count': plenary_count,
                    'parallel_count': parallel_count,
                    'total_count': plenary_count + parallel_count,
                    'plenary_pct': plenary_pct,
                    'parallel_pct': parallel_pct,
                    'ratio': ratio
                }
        
        # Sort countries by ratio
        sorted_countries = sorted(representation_data.items(), 
                                 key=lambda x: x[1]['ratio'], 
                                 reverse=True)
        
        # Limit to top and bottom 15 countries (or all if fewer)
        num_countries = min(30, len(sorted_countries))
        top_countries = sorted_countries[:num_countries]
        
        # Create lists for plotting
        country_names = [c[0] for c in top_countries]
        ratios = [c[1]['ratio'] for c in top_countries]
        total_counts = [c[1]['total_count'] for c in top_countries]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bars
        bars = plt.barh(range(len(country_names)), ratios, height=0.7)
        
        # Color code bars (values > 1 in green, < 1 in red)
        for i, bar in enumerate(bars):
            if ratios[i] > 1.1:
                bar.set_color('#2ca02c')  # Green for overrepresentation
            elif ratios[i] < 0.9:
                bar.set_color('#d62728')  # Red for underrepresentation
            else:
                bar.set_color('#1f77b4')  # Blue for roughly equal
        
        # Add a vertical line at x=1 to indicate equal representation
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        
        # Add text annotations
        for i, (ratio, count) in enumerate(zip(ratios, total_counts)):
            # Format ratio to 2 decimal places
            ratio_text = f"{ratio:.2f}"
            
            # Position the text
            if ratio > 1.1:
                plt.text(ratio + 0.1, i, f"{ratio_text} ({count} talks)", 
                        va='center', ha='left', fontsize=9)
            else:
                plt.text(max(0.1, ratio - 0.5), i, f"{ratio_text} ({count} talks)", 
                        va='center', ha='left', fontsize=9)
        
        # Set plot title and labels
        plt.title('Representation Ratio: Plenary vs. Parallel Talks by Country', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Ratio of % Plenary Talks to % Parallel Talks', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        
        # Set y-axis ticks and labels
        plt.yticks(range(len(country_names)), country_names)
        
        # Add reference text to explain the ratio
        plt.text(0.5, -0.1, 
                "Ratio > 1: Country has proportionally more plenary than parallel talks (overrepresented)\n" +
                "Ratio < 1: Country has proportionally fewer plenary than parallel talks (underrepresented)",
                transform=plt.gca().transAxes, ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='#f9f9f9', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the explanation text
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Representation ratio plot saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating representation ratio plot: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create a simple error plot so at least something is generated
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating representation ratio plot:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, dpi=300)
        plt.close()

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
        
        # Extract sorted years list for use in trending charts
        all_years = [year for year, _ in conferences if year in conference_data]
        
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
        
        # Ensure figures directory exists
        figures_dir = 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # Extract data for trending charts
        plenary_country_data = {}
        parallel_country_data = {}
        plenary_institute_data = {}
        parallel_institute_data = {}
        
        for year in all_years:
            data = conference_data.get(year, {})
            
            # Process country data
            plenary_country_counts = {}
            parallel_country_counts = {}
            
            for talk in data.get('plenary_talks', []):
                country = talk.get('Country', 'Unknown')
                plenary_country_counts[country] = plenary_country_counts.get(country, 0) + 1
                
            for talk in data.get('parallel_talks', []):
                country = talk.get('Country', 'Unknown')
                parallel_country_counts[country] = parallel_country_counts.get(country, 0) + 1
            
            plenary_country_data[year] = plenary_country_counts
            parallel_country_data[year] = parallel_country_counts
            
            # Process institute data
            plenary_institute_counts = {}
            parallel_institute_counts = {}
            
            for talk in data.get('plenary_talks', []):
                institute = talk.get('Institute', 'Unknown')
                plenary_institute_counts[institute] = plenary_institute_counts.get(institute, 0) + 1
                
            for talk in data.get('parallel_talks', []):
                institute = talk.get('Institute', 'Unknown')
                parallel_institute_counts[institute] = parallel_institute_counts.get(institute, 0) + 1
            
            plenary_institute_data[year] = plenary_institute_counts
            parallel_institute_data[year] = parallel_institute_counts
        
        # Create diversity plots
        create_country_trends_plot(all_years, plenary_country_data, parallel_country_data, figures_dir)
        create_institute_bubble_plot(all_years, plenary_institute_data, parallel_institute_data, figures_dir)
        create_regional_diversity_plot(all_years, plenary_country_data, parallel_country_data, figures_dir)
        create_diversity_metrics_plot(all_years, plenary_country_data, parallel_country_data, figures_dir)
        create_representation_ratio_plot(all_years, plenary_country_data, parallel_country_data, figures_dir)
        
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 