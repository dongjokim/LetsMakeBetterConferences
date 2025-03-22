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
import csv
from bs4 import BeautifulSoup
import seaborn as sns
import sys
import time

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

# Define country names and keywords for detection
COUNTRY_NAMES = {
    'United States', 'USA', 'U.S.A.', 'U.S.', 'America', 'United States of America',
    'Germany', 'Deutschland', 
    'France',
    'United Kingdom', 'UK', 'U.K.', 'Britain', 'Great Britain', 'England', 'Scotland', 'Wales',
    'Japan',
    'China', 'P.R. China', 'People\'s Republic of China',
    'Italy', 'Italia',
    'Canada',
    'Russia', 'Russian Federation',
    'India',
    'Brazil',
    'Switzerland', 'Schweiz', 'Suisse',
    'Korea', 'South Korea', 'Republic of Korea',
    'Netherlands', 'The Netherlands', 'Holland',
    'Spain', 'España',
    'Poland', 'Polska',
    'Australia',
    'Sweden', 'Sverige',
    'Finland', 'Suomi',
    'Norway',
    'Denmark',
    'Belgium',
    'Austria',
    'Czech Republic', 'Czechia',
    'Hungary',
    'Portugal',
    'Greece',
    'Israel',
    'Turkey',
    'Mexico',
    'South Africa',
    'Ireland',
    'Romania',
    'Singapore',
    'Ukraine',
    'Taiwan',
    'Bulgaria',
    'Croatia',
    'Slovakia',
    'Slovenia',
    'Serbia',
    'Estonia',
    'Latvia',
    'Lithuania',
    'Argentina',
    'Chile',
    'Colombia',
    'Peru',
    'Venezuela',
    'New Zealand',
    'Egypt',
    'Morocco'
}

# Country keywords for pattern matching
COUNTRY_KEYWORDS = {
    'USA': ['USA', 'United States', 'America', 'U.S.A.', 'U.S.', 'Berkeley', 'MIT', 
            'Brookhaven', 'BNL', 'FNAL', 'Fermilab', 'Los Alamos', 'LANL', 'Argonne', 
            'ANL', 'LBNL', 'ORNL', 'SLAC', 'Chicago', 'Yale', 'Harvard', 'Princeton',
            'Columbia', 'NYU', 'Stony Brook', 'Vanderbilt', 'Ohio', 'Michigan', 'UCLA',
            'Caltech', 'Texas', 'Illinois', 'Indiana', 'Purdue', 'Iowa', 'Maryland',
            'Washington', 'Oregon', 'California', 'Florida', 'Georgia', 'Tennessee',
            'Pennsylvania', 'New York', 'Massachusetts', 'New Jersey', 'Connecticut',
            'Stanford', 'Duke', 'Rutgers', 'Minnesota', 'Colorado', 'Arizona', 'Kansas',
            'Kentucky', 'Alabama', 'Virginia', 'Notre Dame', 'Rice', 'Northwestern'],
    'UK': ['UK', 'United Kingdom', 'Britain', 'England', 'Scotland', 'Wales',
           'Oxford', 'Cambridge', 'Imperial', 'UCL', 'Edinburgh', 'Manchester',
           'Birmingham', 'Liverpool', 'Glasgow', 'Bristol', 'Durham', 'Warwick',
           'King\'s College', 'Queen Mary', 'Nottingham', 'Sheffield', 'Southampton']
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
    'University of Jyvaskyla': 'Finland',
    'Jyvaskyla': 'Finland',
    'Helsinki Institute of Physics': 'Finland',
    'University of Helsinki': 'Finland',
    'Aalto University': 'Finland',
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

# At the beginning of the file, add:
INSTITUTE_COUNTRY_MAPPINGS = {}

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
    """Load institute-to-country mappings from external database file with enhanced matching"""
    database_file = 'institute_country_database.csv'
    institute_country = {}
    normalized_map = {}
    
    # Use the built-in INSTITUTION_COUNTRY mapping as a base
    institute_country.update(INSTITUTION_COUNTRY)
    
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
        'subatech': 'France',
        # Add Finnish institutions explicitly
        'University of Jyvaskyla': 'Finland',
        'University of Jyväskylä': 'Finland',
        'Jyvaskyla': 'Finland',
        'Jyväskylä': 'Finland',
        'Helsinki Institute of Physics': 'Finland',
        'University of Helsinki': 'Finland',
        'HIP': 'Finland',
        'Aalto University': 'Finland',
        'JYFL': 'Finland',
    }
    
    # Add the exact mappings
    institute_country.update(exact_mappings)
    
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
        
        # Also normalize all the exact mappings
        for institute, country in exact_mappings.items():
            normalized = normalize_institute_name(institute)
            if normalized:
                normalized_map[normalized] = country
                
        print(f"Loaded {len(institute_country)} institute-to-country mappings")
    except FileNotFoundError:
        print(f"Warning: Institute-country database file '{database_file}' not found, using built-in mappings")
    
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
    
    # Check if the affiliation directly contains a country name at the end
    # This handles cases like "University of Jyvaskyla, Finland"
    if ',' in affiliation:
        last_part = affiliation.split(',')[-1].strip()
        # Check if this last part is a known country name
        for country, keywords in COUNTRY_KEYWORDS.items():
            if last_part.upper() in [k.upper() for k in keywords]:
                return country
    
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
    """
    Extract speaker name, affiliation, and country from speaker data.
    
    Parameters:
    - speakers: List of speaker data from Indico
    
    Returns:
    - name: Speaker name
    - affiliation: Speaker affiliation
    - country: Speaker country
    """
    name = "Unknown"
    affiliation = "Unknown"
    country = "Unknown"
    
    if not speakers:
        return name, affiliation, country
    
    # Get the first speaker
    speaker = speakers[0]
    
    # Extract name
    if isinstance(speaker, dict):
        # Try different possible keys for name
        for key in ['name', 'fullName', 'full_name', 'person_name', 'title']:
            if key in speaker:
                name = speaker[key]
                break
    
    # Extract affiliation
    if isinstance(speaker, dict):
        # Try different possible keys for affiliation
        for key in ['affiliation', 'institution', 'company', 'organization']:
            if key in speaker and speaker[key]:
                affiliation = speaker[key]
                break
    
    # Extract country from affiliation
    country = extract_country_from_affiliation(affiliation)
    
    return name, affiliation, country

def extract_country_from_affiliation(affiliation):
    """
    Extract country from affiliation string using pattern matching.
    
    Parameters:
    - affiliation: Affiliation string
    
    Returns:
    - country: Extracted country or 'Unknown'
    """
    # Define country keywords for pattern matching with expanded lists
    COUNTRY_KEYWORDS = {
        'USA': ['USA', 'United States', 'America', 'U.S.A.', 'U.S.', 'Berkeley', 'MIT', 
                'Brookhaven', 'BNL', 'FNAL', 'Fermilab', 'Los Alamos', 'LANL', 'Argonne', 
                'ANL', 'LBNL', 'ORNL', 'SLAC', 'Chicago', 'Yale', 'Harvard', 'Princeton',
                'Columbia', 'NYU', 'Stony Brook', 'Vanderbilt', 'Ohio', 'Michigan', 'UCLA',
                'Caltech', 'Texas', 'Illinois', 'Indiana', 'Purdue', 'Iowa', 'Maryland',
                'Washington', 'Oregon', 'California', 'Florida', 'Georgia', 'Tennessee',
                'Pennsylvania', 'New York', 'Massachusetts', 'New Jersey', 'Connecticut',
                'Stanford', 'Duke', 'Rutgers', 'Minnesota', 'Colorado', 'Arizona', 'Kansas',
                'Kentucky', 'Alabama', 'Virginia', 'Notre Dame', 'Rice', 'Northwestern'],
        'UK': ['UK', 'United Kingdom', 'Britain', 'England', 'Scotland', 'Wales',
               'Oxford', 'Cambridge', 'Imperial', 'UCL', 'Edinburgh', 'Manchester',
               'Birmingham', 'Liverpool', 'Glasgow', 'Bristol', 'Durham', 'Warwick',
               'King\'s College', 'Queen Mary', 'Nottingham', 'Sheffield', 'Southampton']
    }
    
    if not affiliation or affiliation == "Unknown":
        return "Unknown"
    
    # Check for direct matches in affiliation name
    for country, keywords in COUNTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in affiliation.lower():
                return country
    
    # Check for common institution patterns
    if 'University' in affiliation or 'Univ.' in affiliation:
        # Extract the university name
        if 'University of' in affiliation:
            university = affiliation.split('University of')[1].strip()
            # Check if the university name contains a country keyword
            for country, keywords in COUNTRY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in university.lower():
                        return country
    
    # Check for common laboratory patterns
    if 'Laboratory' in affiliation or 'Lab' in affiliation or 'National' in affiliation:
        for country, keywords in COUNTRY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in affiliation.lower():
                    return country
    
    # Special cases for major international labs
    if 'CERN' in affiliation:
        return 'Switzerland'
    if 'JINR' in affiliation or 'Dubna' in affiliation:
        return 'Russia'
    if 'GSI' in affiliation:
        return 'Germany'
    if 'KEK' in affiliation:
        return 'Japan'
    
    return 'Unknown'

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


def extract_affiliation_info(talk_data):
    """
    Extract institute information more effectively from talk data.
    Uses global PARTICIPANT_AFFILIATIONS if available.
    Country is determined from institute name consistently.
    
    Parameters:
    - talk_data: Dictionary containing talk information
    
    Returns:
    - Tuple of (institute, country)
    """
    global PARTICIPANT_AFFILIATIONS
    
    institute = talk_data.get('Institute', '').strip()
    speaker = talk_data.get('Speaker', '').strip()
    
    # First try to match with participant data if available
    if 'PARTICIPANT_AFFILIATIONS' in globals() and PARTICIPANT_AFFILIATIONS and speaker:
        # Try exact match first
        if speaker in PARTICIPANT_AFFILIATIONS:
            affiliation = PARTICIPANT_AFFILIATIONS[speaker]
            
            # Update institute if needed
            if (not institute or institute.lower() == 'unknown') and affiliation['Institute'] != 'Unknown':
                institute = affiliation['Institute']
        else:
            # Try fuzzy matching
            speaker_lower = speaker.lower()
            for name, affiliation in PARTICIPANT_AFFILIATIONS.items():
                name_lower = name.lower()
                
                # Check if either name contains the other
                if speaker_lower in name_lower or name_lower in speaker_lower:
                    # Update institute if needed
                    if (not institute or institute.lower() == 'unknown') and affiliation['Institute'] != 'Unknown':
                        institute = affiliation['Institute']
                    break
    
    # Now determine country from institute name
    country = 'Unknown'
    if institute:
        # First check for country code in parentheses
        match = re.search(r'\((..)\)$', institute)
        if match:
            code = match.group(1)
            country = country_map.get(code, 'Unknown')
        
        # If no country code found, try matching with known institutions
        if country == 'Unknown':
            inst_lower = institute.lower()
            for known_inst, known_country in INSTITUTE_COUNTRY_MAPPINGS.items():
                if known_inst.lower() in inst_lower:
                    country = known_country
                    break
    
    return institute, country

def load_indico_ids_from_file(filename='listofQMindigo'):
    """
    Load Indico event IDs from the specified file.
    
    Parameters:
    - filename: Path to file containing Indico IDs
    
    Returns:
    - Dictionary mapping years to Indico IDs
    """
    indico_ids = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines and comments (after removing the comment part)
                content = line.split('#')[0].strip()
                if not content:
                    continue
                
                # Split by whitespace and extract year and ID
                parts = content.split()
                if len(parts) >= 2:
                    year = parts[0]
                    indico_id = parts[1]
                    indico_ids[year] = indico_id
        
        print(f"Loaded {len(indico_ids)} Indico IDs from {filename}")
        return indico_ids
    except Exception as e:
        print(f"Error loading Indico IDs from {filename}: {str(e)}")
        return {}

def extract_participants_from_contributions(indico_id, year, output_file='data/participants.csv'):
    """
    Extract participant information from contribution data in the Indico API.
    
    Parameters:
    - indico_id: ID of the Indico event
    - year: Conference year (for context)
    - output_file: Where to save the participants CSV
    
    Returns:
    - Dictionary mapping participant names to their affiliations
    """
    url = f"https://indico.cern.ch/export/event/{indico_id}.json?detail=contributions&pretty=yes"
    print(f"\nExtracting participant data from contributions API: {url}")
    
    participants = {}
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Fetch data from Indico API
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for error responses
        if 'error' in data:
            print(f"API error: {data['error']}")
            return {}
        
        # Process participant data based on Indico's JSON structure
        results = data.get('results', [])
        if not results:
            print("No data found in the API response.")
            return {}
        
        event_data = results[0]
        contributions = event_data.get('contributions', [])
        
        # Collect all unique participants from contributions
        for contribution in contributions:
            # Extract speakers
            speakers = (
                contribution.get('speakers', []) or 
                contribution.get('person_links', []) or 
                contribution.get('primary_authors', []) or 
                contribution.get('coauthors', []) or
                []
            )
            
            # Process each speaker
            for speaker in speakers:
                # Extract name and affiliation info
                name = (speaker.get('fullName') or 
                       f"{speaker.get('first_name', '')} {speaker.get('last_name', '')}" or 
                       speaker.get('name', '')).strip()
                
                affiliation = speaker.get('affiliation', '')
                country = speaker.get('country', '')
                
                # Skip empty names
                if not name:
                    continue
                
                # Normalize name if possible (to LastName, FirstName format)
                if ',' not in name and ' ' in name:
                    parts = name.split()
                    if len(parts) > 1:
                        normalized_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                    else:
                        normalized_name = name
                else:
                    normalized_name = name
                
                # Store with the normalized name
                if normalized_name not in participants or not participants[normalized_name].get('Institute'):
                    participants[normalized_name] = {
                        'Institute': affiliation,
                        'Country': country,
                        'Year': year,
                        'OriginalName': name
                    }
        
        # Write to CSV file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Name,Institute,Country,Year\n")
            
            for name, info in participants.items():
                # Escape quotes in CSV values
                safe_name = name.replace('"', '""')
                safe_institute = info.get('Institute', '').replace('"', '""')
                safe_country = info.get('Country', '').replace('"', '""')
                
                f.write(f'"{safe_name}","{safe_institute}","{safe_country}","{info.get("Year", "")}"\n')
        
        print(f"Successfully extracted {len(participants)} participants from contribution data")
        return participants
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contribution data: {str(e)}")
        return {}
    except json.JSONDecodeError:
        print("Error parsing JSON response from Indico API")
        return {}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {}

def load_participant_data():
    """Load processed participant data from all_participants.csv file"""
    participant_file = 'data/participants/all_participants.csv'
    
    try:
        # First check if file exists
        if not os.path.exists(participant_file):
            print(f"\nWarning: Participant data file not found at {participant_file}")
            print("Please run 'python QM/fetch_participants.py' first to generate the data.")
            return {}
            
        # Create a lookup dictionary for faster access
        participant_lookup = {}
        years_processed = set()
        total_records = 0
        
        with open(participant_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_records += 1
                name = row['name']
                affiliation = row['affiliation']
                country = row['country']
                year = row['year']
                
                # Extract country from affiliation if not provided
                if not country and '(' in affiliation:
                    match = re.search(r'\((..)\)$', affiliation)
                    if match:
                        country = match.group(1)
                
                # Clean up country codes
                if country:
                    country_map = {
                        'US': 'United States',
                        'USA': 'United States',
                        'UK': 'United Kingdom',
                        'AT': 'Austria',
                        'PL': 'Poland',
                        'DE': 'Germany',
                        'FR': 'France',
                        'IT': 'Italy',
                        'JP': 'Japan',
                        'CN': 'China',
                        'IN': 'India',
                        'CH': 'Switzerland',
                        'NL': 'Netherlands',
                        'RU': 'Russia',
                        'BR': 'Brazil',
                        'ES': 'Spain'
                    }
                    country = country_map.get(country, country)
                
                # Detect country from known institutions if still unknown
                if not country:
                    inst_lower = affiliation.lower()
                    inst_country_map = {
                        'columbia university': 'United States',
                        'wayne state': 'United States',
                        'ohio university': 'United States',
                        'university of tennessee': 'United States',
                        'stony brook': 'United States',
                        'brookhaven': 'United States',
                        'agh university': 'Poland',
                        'nuclear physics polish': 'Poland',
                        'austrian academy': 'Austria'
                    }
                    
                    for inst, inst_country in inst_country_map.items():
                        if inst in inst_lower:
                            country = inst_country
                            break
                
                years_processed.add(year)
                
                # Store both original and normalized versions of the name
                participant_data = {
                    'affiliation': affiliation,
                    'country': country if country else 'Unknown',
                    'year': year
                }
                
                participant_lookup[name] = participant_data
                
                # Also store last_name, first_name format if possible
                if ',' not in name and ' ' in name:
                    parts = name.split()
                    if len(parts) > 1:
                        normalized_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                        participant_lookup[normalized_name] = participant_data
        
        print(f"\nLoaded {len(participant_lookup)} participant records from {total_records} entries")
        print(f"Years covered: {sorted(years_processed)}")
        
        # Print statistics about unknown countries
        unknown_count = sum(1 for p in participant_lookup.values() if p['country'] == 'Unknown')
        if unknown_count > 0:
            print(f"\nWarning: {unknown_count} participants still have unknown countries")
            print("\nSample of entries with unknown countries:")
            for name, data in participant_lookup.items():
                if data['country'] == 'Unknown':
                    print(f"  {name}: {data['affiliation']}")
                    if len(data['affiliation']) > 0:  # Only show a few examples
                        break
        
        return participant_lookup
    
    except Exception as e:
        print(f"Error loading participant data: {e}")
        import traceback
        traceback.print_exc()
        return {}

def fix_unknown_institutes_from_participants(conference_data, participant_lookup):
    """
    Fix unknown institutes and countries for speakers using participant data.
    
    Parameters:
    - conference_data: Dictionary with conference data
    - participant_lookup: Dictionary mapping names to participant info
    
    Returns:
    - Tuple of (institute_fixes, country_fixes)
    """
    print("\nFixing unknown institutes using participant data...")
    
    institute_fixes = 0
    country_fixes = 0
    
    for year, data in conference_data.items():
        # Process all talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type not in data:
                continue
                
            for talk in data[talk_type]:
                speaker = talk.get('Speaker', '')
                if not speaker:
                    continue
                
                # Try exact match first
                participant = participant_lookup.get(speaker)
                
                # If no exact match, try fuzzy matching
                if not participant:
                    speaker_lower = speaker.lower()
                    for p_name, p_data in participant_lookup.items():
                        if speaker_lower in p_name.lower() or p_name.lower() in speaker_lower:
                            participant = p_data
                            break
                
                if participant:
                    # Fix institute if unknown
                    if not talk.get('Institute') or talk['Institute'] == 'Unknown':
                        talk['Institute'] = participant['affiliation']
                        institute_fixes += 1
                    
                    # Fix country if unknown
                    if not talk.get('Country') or talk['Country'] == 'Unknown':
                        talk['Country'] = participant['country']
                        country_fixes += 1
    
    print(f"Applied {institute_fixes} institute fixes and {country_fixes} country fixes")
    return institute_fixes, country_fixes

def save_processed_data(conference_data, output_dir='data/processed'):
    """Save processed conference data to CSV files"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert data to DataFrames and save by year and type
        for year, data in conference_data.items():
            year_dir = os.path.join(output_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            
            # Save different types of talks
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks', 'other_talks']:
                if talk_type in data and data[talk_type]:
                    df = pd.DataFrame(data[talk_type])
                    output_file = os.path.join(year_dir, f'{talk_type}.csv')
                    df.to_csv(output_file, index=False)
                    print(f"Saved {len(df)} {talk_type} to {output_file}")
            
            # Save all talks combined
            if 'all_talks' in data and data['all_talks']:
                df = pd.DataFrame(data['all_talks'])
                output_file = os.path.join(year_dir, 'all_talks.csv')
                df.to_csv(output_file, index=False)
                print(f"Saved {len(df)} total talks to {output_file}")
            
            # Save statistics
            stats = {
                'total_main': data.get('total_main', 0),
                'unknown_affiliations': data.get('unknown_affiliations', {})
            }
            stats_file = os.path.join(year_dir, 'statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_file}")
        
        print("\nAll data saved successfully")
        
    except Exception as e:
        print(f"Error saving processed data: {e}")
        import traceback
        traceback.print_exc()

def fetch_and_analyze_conferences():
    """Main function to fetch and analyze conference data."""
    try:
        # Load participant data first
        print("\nLoading participant data...")
        participant_lookup = load_participant_data()
        
        # Rest of your existing code...
        indico_ids = load_indico_ids_from_file('listofQMindigo')
        
        # Process each conference
        conferences = []
        for year, indico_id in indico_ids.items():
            conferences.append((year, indico_id))
        
        # Sort conferences by year
        conferences.sort(key=lambda x: x[0])
        
        # Process each conference
        conference_data = {}
        
        for year, indico_id in conferences:
            print(f"\nProcessing QM{year} (Indico ID: {indico_id})...")
            data = fetch_and_process_contributions(indico_id, year)
            if data:
                conference_data[year] = data
        
        # Save processed data
        save_processed_data(conference_data)
        
        return conference_data
        
    except Exception as e:
        print(f"Error in fetch_and_analyze_conferences: {str(e)}")
        import traceback
        traceback.print_exc()

def fix_common_affiliation_problems(conference_data):
    """Fix known problematic affiliations"""
    print("\nFixing common affiliation problems...")
    
    fixes_applied = 0
    
    # Known problematic affiliations and their correct countries
    problem_fixes = {
        'University of Jyvaskyla': 'Finland',
        'Jyvaskyla University': 'Finland',
        'University of Helsinki': 'Finland',
        'Helsinki Institute of Physics': 'Finland',
        'JYFL': 'Finland',  # Jyväskylän yliopiston fysiikan laitos (Department of Physics, University of Jyväskylä)
        # Add more problematic institutions as needed
    }
    
    # Process all talks in all conferences
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type not in data:
                continue
                
            for talk in data[talk_type]:
                institute = talk.get('Institute', '')
                # Check if this is one of our problematic cases
                for problem_inst, correct_country in problem_fixes.items():
                    if problem_inst.lower() in institute.lower():
                        if talk.get('Country') == 'Unknown' or talk.get('Country') != correct_country:
                            talk['Country'] = correct_country
                            fixes_applied += 1
                    
                # Also directly check for ", Finland" in the institute name
                if ', Finland' in institute and talk.get('Country') == 'Unknown':
                    talk['Country'] = 'Finland'
                    fixes_applied += 1
                    
    print(f"Applied {fixes_applied} fixes for common affiliation problems")
    return fixes_applied

def print_unknown_institute_examples(conference_data, num_examples=3):
    """Print examples of talks with unknown institutes"""
    print("\n===== EXAMPLES OF TALKS WITH UNKNOWN INSTITUTES =====")
    
    # Print examples for each year
    for year, data in sorted(conference_data.items()):
        # Check plenary talks
        unknown_plenary = [t for t in data.get('plenary_talks', []) if t.get('Institute', '') == 'Unknown']
        unknown_parallel = [t for t in data.get('parallel_talks', []) if t.get('Institute', '') == 'Unknown']
        
        if unknown_plenary or unknown_parallel:
            print(f"\nYear: {year}")
            
            if unknown_plenary:
                print(f"  Plenary talks with unknown institutes: {len(unknown_plenary)}")
                for i, talk in enumerate(unknown_plenary[:num_examples]):
                    print(f"    Example {i+1}: {talk.get('Speaker', 'No name')} - {talk.get('Title', 'No title')[:50]}...")
            
            if unknown_parallel:
                print(f"  Parallel talks with unknown institutes: {len(unknown_parallel)}")
                for i, talk in enumerate(unknown_parallel[:num_examples]):
                    print(f"    Example {i+1}: {talk.get('Speaker', 'No name')} - {talk.get('Title', 'No title')[:50]}...")


def should_reprocess_data(max_age_days=1):
    """
    Check if we should reprocess the data based on the age of the processed data file.
    
    Parameters:
    - max_age_days: Maximum age in days for the processed data file
    
    Returns:
    - True if data should be reprocessed, False otherwise
    """
    processed_file = 'data/processed_conference_data.json'
    
    # If file doesn't exist, we need to process
    if not os.path.exists(processed_file):
        print(f"Processed data file '{processed_file}' not found. Will process data.")
        return True
    
    # Check file age
    file_time = os.path.getmtime(processed_file)
    current_time = time.time()
    file_age_days = (current_time - file_time) / (60 * 60 * 24)  # Convert seconds to days
    
    if file_age_days <= max_age_days:
        print(f"Processed data file is {file_age_days:.1f} days old (max: {max_age_days} days). Using existing data.")
        return False
    else:
        print(f"Processed data file is {file_age_days:.1f} days old (max: {max_age_days} days). Will reprocess data.")
        return True

def print_summary_table(conference_data, title="Conference Summary"):
    """
    Print a summary table of conference data.
    
    Parameters:
    - conference_data: Dictionary with conference data
    - title: Title for the summary table
    """
    print(f"\n{title}")
    print("Year Location                   Total  Plenary  Parallel Poster Flash Unk_Plen Unk_Par")
    print("-" * 85)
    
    # Extract sorted years list
    all_years = sorted(conference_data.keys())
    
    for year in all_years:
        data = conference_data[year]
        location = CONFERENCE_LOCATIONS.get(year, 'Unknown location')
        total = len(data['all_talks'])
        plenary = len(data['plenary_talks'])
        parallel = len(data['parallel_talks'])
        poster = len(data['poster_talks'])
        flash = FLASH_TALK_COUNTS.get(year, 0)
        
        # Count unknown institutes
        unknown_plenary = len([t for t in data['plenary_talks'] if t.get('Institute', '') == 'Unknown'])
        unknown_parallel = len([t for t in data['parallel_talks'] if t.get('Institute', '') == 'Unknown'])
        
        print(f"{year} {location:<25} {total:<6} {plenary:<8} {parallel:<8} {poster:<6} {flash:<5} {unknown_plenary:<8} {unknown_parallel}")

# Update the main section to call this function before and after updating speaker info
if __name__ == "__main__":
    try:
        # Import time module at the top of the file
        import time
        
        # Check if we should reprocess the data
        if not should_reprocess_data(max_age_days=1):
            # Load existing processed data
            try:
                with open('data/processed_conference_data.json', 'r') as f:
                    conference_data = json.load(f)
                print(f"Loaded processed data for {len(conference_data)} conferences.")
                
                # Print initial summary table
                print_summary_table(conference_data, "Initial Conference Summary")
                
                # Load participant data
                print("\nLoading participant data...")
                participant_data = load_participant_data()
                
                # Update speaker information from participant data
                if participant_data:
                    fix_unknown_institutes_from_participants(conference_data, participant_data)
                    
                    # Print updated summary table
                    print_summary_table(conference_data, "Conference Summary After Updates")
                    
                    # Save the updated data
                    with open('data/processed_conference_data.json', 'w') as f:
                        json.dump(conference_data, f, indent=2)
                    print("Saved updated conference data.")
                
                # Print examples of talks with unknown institutes
                print_unknown_institute_examples(conference_data)
                
                sys.exit(0)  # Exit successfully
            except Exception as e:
                print(f"Error loading processed data: {e}")
                print("Will process data from scratch.")
        
        # If we get here, we need to process the data
        with open('listofQMindigo', 'r') as f:
            conferences = [line.strip().split()[:2] for line in f if not line.strip().startswith('#')]
            
        conference_data = {}
        
        # Sort conferences by year
        conferences.sort(key=lambda x: x[0])
        
        # Process each conference
        for year, indico_id in conferences:
            print(f"\nProcessing QM{year} (Indico ID: {indico_id})...")
            data = fetch_and_process_contributions(indico_id, year)
            if data:
                conference_data[year] = data
        
        # Print initial summary table
        print_summary_table(conference_data, "Initial Conference Summary")
        
        # Load participant data
        print("\nLoading participant data...")
        participant_data = load_participant_data()
        
        # Update speaker information from participant data
        if participant_data:
            fix_unknown_institutes_from_participants(conference_data, participant_data)
            
            # Print updated summary table
            print_summary_table(conference_data, "Conference Summary After Updates")
        
        # Print examples of talks with unknown institutes
        print_unknown_institute_examples(conference_data)
        
        # Save the processed data
        save_processed_data(conference_data)
        with open('data/processed_conference_data.json', 'w') as f:
            json.dump(conference_data, f, indent=2)
        print("Saved processed conference data.")
        
        # Count remaining unknown institutes and countries
        total_talks = 0
        unknown_institutes = 0
        unknown_countries = 0
        
        for year, data in conference_data.items():
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                if talk_type not in data:
                    continue
                    
                talks = data[talk_type]
                total_talks += len(talks)
                
                for talk in talks:
                    if talk.get('Institute', '') == 'Unknown':
                        unknown_institutes += 1
                    if talk.get('Country', '') == 'Unknown':
                        unknown_countries += 1
        
        print("\n===== FINAL STATISTICS =====")
        print(f"Total talks processed: {total_talks}")
        print(f"Remaining unknown institutes: {unknown_institutes} ({unknown_institutes/total_talks*100:.1f}%)")
        print(f"Remaining unknown countries: {unknown_countries} ({unknown_countries/total_talks*100:.1f}%)")
        
        # Print breakdown by talk type
        print("\nBreakdown by talk type:")
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            type_total = 0
            type_unknown_inst = 0
            
            for year, data in conference_data.items():
                if talk_type in data:
                    talks = data[talk_type]
                    type_total += len(talks)
                    type_unknown_inst += sum(1 for t in talks if t.get('Institute', '') == 'Unknown')
            
            if type_total > 0:
                print(f"  {talk_type.replace('_', ' ').title()}: {type_unknown_inst}/{type_total} unknown institutes ({type_unknown_inst/type_total*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def generate_final_statistics_report(conference_data, output_file='final_statistics_report.txt'):
    """
    Generate a final report of statistics including speakers without institute or country.
    """
    print("\nGenerating final statistics report...")
    
    # Initialize counters
    total_speakers = 0
    unknown_institute_count = 0
    unknown_country_count = 0
    unknown_both_count = 0
    unknown_speakers = []
    
    # Count by year and talk type
    year_stats = {}
    
    try:
        # Open the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FINAL STATISTICS REPORT\n")
            f.write("======================\n\n")
            
            # Process each conference year
            for year in sorted(conference_data.keys()):
                data = conference_data[year]
                
                year_total = 0
                year_unknown_institute = 0
                year_unknown_country = 0
                year_unknown_both = 0
                
                # Process each talk type
                for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
                    if talk_type not in data:
                        continue
                    
                    talks = data[talk_type]
                    if not talks:  # Skip if empty
                        continue
                        
                    type_total = len(talks)
                    type_unknown_institute = 0
                    type_unknown_country = 0
                    type_unknown_both = 0
                    
                    # Check each talk - use exact string comparison
                    for talk in talks:
                        institute = talk.get('Institute', '')
                        country = talk.get('Country', '')
                        speaker = talk.get('Speaker', '')
                        
                        # Count unknowns - be very explicit
                        institute_unknown = (institute == 'Unknown')
                        country_unknown = (country == 'Unknown')
                        
                        if institute_unknown:
                            type_unknown_institute += 1
                        
                        if country_unknown:
                            type_unknown_country += 1
                        
                        if institute_unknown and country_unknown:
                            type_unknown_both += 1
                            unknown_speakers.append({
                                'Year': year,
                                'Type': talk_type,
                                'Speaker': speaker,
                                'Title': talk.get('Title', 'No Title')
                            })
                    
                    # Update counters
                    year_total += type_total
                    year_unknown_institute += type_unknown_institute
                    year_unknown_country += type_unknown_country
                    year_unknown_both += type_unknown_both
                    
                    # Write talk type statistics
                    type_label = talk_type.replace('_', ' ').title()
                    f.write(f"{year} - {type_label}:\n")
                    f.write(f"  Total: {type_total}\n")
                    f.write(f"  Unknown Institute: {type_unknown_institute}\n")
                    f.write(f"  Unknown Country: {type_unknown_country}\n")
                    f.write(f"  Unknown Both: {type_unknown_both}\n\n")
                
                # Update total counters
                total_speakers += year_total
                unknown_institute_count += year_unknown_institute
                unknown_country_count += year_unknown_country
                unknown_both_count += year_unknown_both
                
                # Store year statistics
                year_stats[year] = {
                    'Total': year_total,
                    'Unknown Institute': year_unknown_institute,
                    'Unknown Country': year_unknown_country,
                    'Unknown Both': year_unknown_both
                }
                
                # Write year summary
                f.write(f"Summary for {year}:\n")
                f.write(f"  Total: {year_total}\n")
                f.write(f"  Unknown Institute: {year_unknown_institute}\n")
                f.write(f"  Unknown Country: {year_unknown_country}\n")
                f.write(f"  Unknown Both: {year_unknown_both}\n")
                f.write("=" * 50 + "\n\n")
            
            # Write overall summary
            f.write("\nOVERALL SUMMARY\n")
            f.write("===============\n")
            f.write(f"Total Speakers: {total_speakers}\n")
            f.write(f"Unknown Institute: {unknown_institute_count}\n")
            f.write(f"Unknown Country: {unknown_country_count}\n")
            f.write(f"Unknown Both: {unknown_both_count}\n\n")
            
            # List speakers with unknown both
            if unknown_speakers:
                f.write("\nSPEAKERS WITH UNKNOWN INSTITUTE AND COUNTRY\n")
                f.write("=========================================\n\n")
                
                for speaker_info in unknown_speakers:
                    f.write(f"Year: {speaker_info['Year']}\n")
                    f.write(f"Type: {speaker_info['Type'].replace('_', ' ').title()}\n")
                    f.write(f"Speaker: {speaker_info['Speaker']}\n")
                    f.write(f"Title: {speaker_info['Title']}\n\n")
        
        print(f"Final statistics report generated: {output_file}")
        print(f"Total speakers: {total_speakers}")
        print(f"Unknown institute: {unknown_institute_count}")
        print(f"Unknown country: {unknown_country_count}")
        print(f"Unknown both: {unknown_both_count}")
        
    except Exception as e:
        print(f"Error generating statistics report: {e}")
    
    return unknown_speakers

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
        flash = len(data.get('flash_talks', [])) or FLASH_TALK_COUNTS.get(year, 0)
        
        # Count unknown institutes - be explicit about what "unknown" means
        unknown_plenary = sum(1 for t in data.get('plenary_talks', []) if t.get('Institute', '') == 'Unknown')
        unknown_parallel = sum(1 for t in data.get('parallel_talks', []) if t.get('Institute', '') == 'Unknown')
        
        print(f"{year} {location:<25} {total:<6} {plenary:<8} {parallel:<8} {poster:<6} {flash:<5} {unknown_plenary:<8} {unknown_parallel}")

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

# In the fetch_and_analyze_conferences function, add debugging:

# Debug the conference_data before trying to print summary
debug_conference_data(conference_data, "Before Summary")

# Try a very simple summary that doesn't assume any structure
print("\nSimple Conference Summary:")
print("Type of conference_data:", type(conference_data))

if isinstance(conference_data, dict):
    for year, data in sorted(conference_data.items()):
        print(f"Year: {year}, Type of data: {type(data)}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
        else:
            print(f"  Value: {data}")
elif isinstance(conference_data, list):
    print(f"List length: {len(conference_data)}")
    for i, item in enumerate(conference_data[:5]):  # Show first 5 items
        print(f"Item {i}: {type(item)}")
        if isinstance(item, tuple) and len(item) >= 2:
            print(f"  Year: {item[0]}, Indico ID: {item[1]}")
else:
    print(f"Unexpected type: {type(conference_data)}")

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

def detect_country(affiliation):
    """Detect country from affiliation string"""
    if not affiliation:
        return 'Unknown'
    
    # Convert to lowercase for case-insensitive matching
    affiliation_lower = affiliation.lower()
    
    # Common university/institute patterns with their countries
    INSTITUTION_PATTERNS = {
        'columbia university': 'United States',
        'wayne state': 'United States',
        'bhabha atomic': 'India',
        'calcutta': 'India',
        'bhubaneswar': 'India',
        'warsaw': 'Poland',
        'polish academy': 'Poland',
        'darmstadt': 'Germany',
        'theoretische physik': 'Germany',
        'institut für': 'Germany',
    }
    
    # First check for country codes in parentheses at the end
    match = re.search(r'\((..)\)$', affiliation)
    if match:
        country_code = match.group(1)
        code_map = {
            'US': 'United States',
            'UK': 'United Kingdom',
            'PL': 'Poland',
            'IT': 'Italy',
            'AT': 'Austria',
            'DE': 'Germany',
            'FR': 'France',
            'JP': 'Japan',
            'CN': 'China',
            'IN': 'India',
            'CH': 'Switzerland',
            'NL': 'Netherlands',
            'RU': 'Russia',
            'BR': 'Brazil',
            'ES': 'Spain'
        }
        if country_code in code_map:
            return code_map[country_code]
    
    # Check institution patterns
    for pattern, country in INSTITUTION_PATTERNS.items():
        if pattern in affiliation_lower:
            return country
    
    # Check for explicit country mentions
    for country in COUNTRY_NAMES:
        if country.lower() in affiliation_lower:
            return country
    
    # Check for specific keywords
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(keyword.lower() in affiliation_lower for keyword in keywords):
            return country
    
    return 'Unknown'

def load_institute_mappings():
    """Load or create institute-to-country mappings file"""
    mappings_file = 'data/unknown_institute_mappings.csv'
    
    # Additional mappings to add if not present
    NEW_MAPPINGS = {
        # German institutions
        'bielefeld': 'Germany',
        'bonn': 'Germany',
        'darmstadt': 'Germany',
        'freiburg': 'Germany',
        'heidelberg': 'Germany',
        'münster': 'Germany',
        'tübingen': 'Germany',
        'wuppertal': 'Germany',
        
        # US institutions
        'bnl': 'United States',
        'lbnl': 'United States',
        'llnl': 'United States',
        'ornl': 'United States',
        'purdue': 'United States',
        'rutgers': 'United States',
        'vanderbilt': 'United States',
        
        # Japanese institutions
        'hiroshima': 'Japan',
        'nagoya': 'Japan',
        'tohoku': 'Japan',
        'waseda': 'Japan',
        
        # Chinese institutions
        'fudan': 'China',
        'huzhou': 'China',
        'lanzhou': 'China',
        'peking': 'China',
        'tsinghua': 'China',
        'wuhan': 'China',
        
        # Indian institutions
        'aligarh': 'India',
        'banaras': 'India',
        'calcutta': 'India',
        'jammu': 'India',
        'panjab': 'India',
        
        # European institutions
        'coimbra': 'Portugal',
        'krakow': 'Poland',
        'leuven': 'Belgium',
        'padova': 'Italy',
        'paris sud': 'France',
        'roma': 'Italy',
        'torino': 'Italy',
        'trento': 'Italy',
        'zurich': 'Switzerland'
    }
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mappings_file), exist_ok=True)
        
        # Load existing mappings
        mappings = {}
        if os.path.exists(mappings_file):
            with open(mappings_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mappings[row['Institute'].lower()] = row['Country']
        
        # Add any new mappings not already present
        new_entries = False
        for inst, country in NEW_MAPPINGS.items():
            if inst.lower() not in mappings:
                mappings[inst.lower()] = country
                new_entries = True
        
        # If we added new entries, write the updated mappings back to file
        if new_entries:
            print("Adding new institute mappings...")
            with open(mappings_file, 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Institute', 'Country'])
                for inst, country in sorted(mappings.items()):
                    writer.writerow([inst, country])
        
        print(f"Loaded {len(mappings)} institute mappings")
        return mappings
    
    except Exception as e:
        print(f"Error handling institute mappings: {e}")
        return {k.lower(): v for k, v in NEW_MAPPINGS.items()}

# Add at the start of the script
INSTITUTE_COUNTRY_MAPPINGS = load_institute_mappings()



