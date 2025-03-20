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
    'Morocco',
    'Tunisia',
    'Nigeria',
    'Kenya',
    'Ethiopia',
    'Ghana',
    'Senegal'
}

# Country keywords for pattern matching
COUNTRY_KEYWORDS = {
    'USA': ['USA', 'United States', 'America', 'U.S.A.', 'U.S.'], 
    'UK': ['UK', 'United Kingdom', 'Britain', 'England', 'Scotland', 'Wales'],
    'Germany': ['Germany', 'Deutschland'],
    'France': ['France'],
    'Japan': ['Japan'],
    'China': ['China', 'P.R. China'],
    'Italy': ['Italy', 'Italia'],
    'Russia': ['Russia', 'Russian Federation'],
    'Switzerland': ['Switzerland', 'Schweiz', 'Suisse'],
    'Finland': ['Finland', 'Suomi'],
    'Korea': ['Korea', 'South Korea'],
    'Canada': ['Canada'],
    'Poland': ['Poland', 'Polska'],
    'Czech Republic': ['Czech Republic', 'Czechia'],
    'South Africa': ['South Africa'],
    'Brazil': ['Brazil'],
    'India': ['India']
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
    """Analyze trends across multiple conferences with improved country resolution"""
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
    
    # Track percentage of unknown affiliations
    plt.figure(figsize=(12, 6))
    unknown_percentages = []
    for year in years:
        all_talks = (conference_data[year].get('plenary_talks', []) + 
                   conference_data[year].get('parallel_talks', []) + 
                   conference_data[year].get('poster_talks', []))
        
        unknown_count = sum(1 for talk in all_talks if talk.get('Country') == 'Unknown')
        total_count = len(all_talks)
        
        if total_count > 0:
            unknown_percentages.append((unknown_count / total_count) * 100)
        else:
            unknown_percentages.append(0)
    
    plt.plot(years, unknown_percentages, 'o-', color='red')
    plt.title('Percentage of Talks with Unknown Country Affiliation')
    plt.xlabel('QM Year')
    plt.ylabel('Percentage (%)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.savefig('figs/trends/unknown_affiliation_rate.pdf', bbox_inches='tight')
    
    # Analyze top countries across years (with improved affiliation handling)
    # Get all unique countries
    all_countries = set()
    for year in years:
        # Only count non-Unknown countries
        country_counts = {k: v for k, v in conference_data[year]['country_counts'].items() 
                         if k != 'Unknown'}
        all_countries.update(country_counts.keys())
    
    # Select top countries overall for tracking
    country_total = Counter()
    for year in years:
        # Filter out Unknown countries from the counts
        country_counts = {k: v for k, v in conference_data[year]['country_counts'].items() 
                         if k != 'Unknown'}
        country_total.update(country_counts)
    
    top_countries = [country for country, _ in country_total.most_common(8)]
    
    # Add a few emerging countries for trend analysis
    emerging_countries = ['Brazil', 'Poland', 'Czech Republic', 'South Africa']
    emerging_in_data = [c for c in emerging_countries if c in all_countries and c not in top_countries]
    
    # Plot trends for top countries
    plt.figure(figsize=(14, 8))
    
    # Plot top countries with solid lines
    for country in top_countries:
        country_by_year = [conference_data[year]['country_counts'].get(country, 0) for year in years]
        plt.plot(years, country_by_year, 'o-', linewidth=2, label=country)
    
    # Plot emerging countries with dashed lines
    for country in emerging_in_data:
        country_by_year = [conference_data[year]['country_counts'].get(country, 0) for year in years]
        plt.plot(years, country_by_year, 'o--', linewidth=1.5, label=f"{country} (Emerging)")
    
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
    
    print("\nData quality metrics:")
    for year in years:
        all_talks = (conference_data[year].get('plenary_talks', []) + 
                   conference_data[year].get('parallel_talks', []) + 
                   conference_data[year].get('poster_talks', []))
        
        unknown_count = sum(1 for talk in all_talks if talk.get('Country') == 'Unknown')
        total_count = len(all_talks)
        
        if total_count > 0:
            unknown_percentage = (unknown_count / total_count) * 100
            print(f"QM{year}: {unknown_percentage:.1f}% unknown country affiliations ({unknown_count}/{total_count})")
        else:
            print(f"QM{year}: No talks data available")

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
    """Print summary of all conferences with improved affiliation resolution metrics"""
    print("\nSummary of all conferences:")
    print("Year  Location                  Total  Plenary  Parallel  Poster  Flash  Unknown_Aff%  Resolution")
    print("-" * 85)
    
    # Prepare data for the plot
    years = []
    top_keywords = []
    top_counts = []
    
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
        
        # Count unknown affiliations and calculate percentage
        all_talks = (conf.get('plenary_talks', []) + 
                   conf.get('parallel_talks', []) + 
                   conf.get('poster_talks', []) + 
                   conf.get('flash_talks', []))
        
        unknown_aff = sum(1 for talk in all_talks if talk.get('Country') == 'Unknown')
        unknown_percentage = (unknown_aff / total) * 100 if total > 0 else 0
        
        # Determine resolution quality
        if unknown_percentage < 3:
            resolution = "Excellent"
        elif unknown_percentage < 7:
            resolution = "Good"
        elif unknown_percentage < 15:
            resolution = "Fair"
        else:
            resolution = "Poor"
        
        print(f"{year}  {location:<22} {total:>6}  {plenary:>7}  {parallel:>8}  {poster:>6}  {flash:>5}  {unknown_percentage:>9.1f}%  {resolution}")
        
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

def plot_top_entities(counts, entity_type, output_prefix, top_n=20):
    """
    Create a horizontal bar chart showing the top entities (countries or institutes).
    
    Parameters:
    - counts: Counter object with entity counts
    - entity_type: String describing the entity type (e.g., 'Countries', 'Institutes')
    - output_prefix: Prefix for the output filename
    - top_n: Number of top entities to show
    """
    # Get the top entities
    top_entities = counts.most_common(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_entities))
    
    # Calculate total for percentages
    total = sum(counts.values())
    
    # Extract names and counts
    names = [item[0] for item in top_entities]
    values = [item[1] for item in top_entities]
    
    # Create horizontal bars
    bars = plt.barh(y_pos, values, align='center', color='skyblue')
    
    # Add percentages to the end of each bar
    for i, (entity, count) in enumerate(top_entities):
        percentage = (count / total) * 100
        plt.text(count + 0.5, i, f"{percentage:.1f}%", va='center')
    
    plt.yticks(y_pos, names)
    plt.xlabel('Number of Contributions')
    plt.title(f'Top {top_n} {entity_type} by Contribution Count')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figures/top_{output_prefix}s.pdf', bbox_inches='tight')
    
    # Print the top entities
    print(f"\nTop {top_n} {entity_type}:")
    for i, (entity, count) in enumerate(top_entities, 1):
        percentage = (count / total) * 100
        print(f"{i}. {entity}: {count} ({percentage:.1f}%)")

def create_country_institute_plots(conference_data):
    """Create plots for country and institute statistics using enhanced mapping"""
    print("\nCreating country and institute plots with enhanced mapping...")
    
    # Make sure output directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Combine data across all conferences
    country_counts = Counter()
    institute_counts = Counter()
    plenary_country = Counter()
    parallel_country = Counter()
    poster_country = Counter()
    
    for year, data in conference_data.items():
        # Ensure we're using the latest country data after all fixes
        if 'plenary_talks' in data:
            for talk in data['plenary_talks']:
                country = talk.get('Country', 'Unknown')
                institute = talk.get('Institute', 'Unknown')
                
                # Skip entries with unknown country or institute
                if country != 'Unknown':
                    country_counts[country] += 1
                    plenary_country[country] += 1
                
                if institute != 'Unknown':
                    institute_counts[institute] += 1
        
        if 'parallel_talks' in data:
            for talk in data['parallel_talks']:
                country = talk.get('Country', 'Unknown')
                institute = talk.get('Institute', 'Unknown')
                
                # Skip entries with unknown country or institute
                if country != 'Unknown':
                    country_counts[country] += 1
                    parallel_country[country] += 1
                
                if institute != 'Unknown':
                    institute_counts[institute] += 1
        
        if 'poster_talks' in data:
            for talk in data['poster_talks']:
                country = talk.get('Country', 'Unknown')
                institute = talk.get('Institute', 'Unknown')
                
                # Skip entries with unknown country or institute
                if country != 'Unknown':
                    country_counts[country] += 1
                    poster_country[country] += 1
                
                if institute != 'Unknown':
                    institute_counts[institute] += 1
    
    # Plot top countries
    plot_top_entities(country_counts, 'Countries', 'country', top_n=20)
    
    # Plot top institutes
    plot_top_entities(institute_counts, 'Institutes', 'institute', top_n=30)
    
    # Create plenary talk distribution by country
    create_plenary_country_plot(plenary_country)
    
    # Create plenary vs parallel representation ratio
    create_representation_ratio_plot(plenary_country, parallel_country)
    
    # Create regional diversity plots
    create_regional_diversity_plot(country_counts, conference_data)
    
    # Create institute bubble chart
    create_institute_bubble_chart(institute_counts, 30)

def create_yearly_country_analysis(conference_data):
    """Analyze and visualize country participation over years with enhanced mapping"""
    print("\nAnalyzing country participation by year...")
    
    # Make sure output directory exists
    os.makedirs('figures', exist_ok=True)
    
    years = sorted([year for year in conference_data.keys() if year.isdigit()])
    
    # Count countries for each year
    country_by_year = {}
    for year in years:
        data = conference_data[year]
        
        # Initialize or reset country counts
        country_counts = Counter()
        
        # Process all talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    country = talk.get('Country', 'Unknown')
                    # Only count if country is known after all mappings and fixes
                    if country != 'Unknown':
                        country_counts[country] += 1
        
        country_by_year[year] = country_counts
    
    # Calculate diversity metrics
    unique_countries_by_year = [len(country_by_year[year]) for year in years]
    
    # Calculate Herfindahl-Hirschman Index (HHI) for each year
    # Lower HHI = more diverse
    hhi_by_year = []
    for year in years:
        counts = country_by_year[year]
        total = sum(counts.values())
        
        if total > 0:
            # Sum of squared market shares
            hhi = sum((count / total * 100) ** 2 for count in counts.values())
            hhi_by_year.append(hhi)
        else:
            hhi_by_year.append(0)
    
    # Create diversity metrics plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot unique countries
    ax1.plot(years, unique_countries_by_year, 'o-', linewidth=2, color='blue')
    ax1.set_ylabel('Number of Unique Countries')
    ax1.set_title('Country Diversity in Quark Matter Conferences')
    ax1.grid(True)
    
    # Annotate values
    for i, count in enumerate(unique_countries_by_year):
        ax1.annotate(str(count), (years[i], count), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Plot HHI
    ax2.plot(years, hhi_by_year, 's-', linewidth=2, color='red')
    ax2.set_ylabel('Herfindahl-Hirschman Index')
    ax2.set_xlabel('Year')
    ax2.grid(True)
    ax2.set_title('Concentration of Contributions (Lower HHI = More Diverse)')
    
    # Annotate values
    for i, hhi in enumerate(hhi_by_year):
        ax2.annotate(f"{hhi:.0f}", (years[i], hhi), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/diversity_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    # Create table with most common countries each year
    top_n = 10
    top_countries_all_time = [country for country, _ in Counter().most_common(top_n)]
    
    # Find the top countries across all years
    all_time_counts = Counter()
    for year_counts in country_by_year.values():
        all_time_counts.update(year_counts)
    
    top_countries_all_time = [country for country, _ in all_time_counts.most_common(top_n)]
    
    # Create a table with years in columns and countries in rows
    table_data = []
    for country in top_countries_all_time:
        row = [country]
        for year in years:
            row.append(country_by_year[year].get(country, 0))
        table_data.append(row)
    
    # Print the table
    print("\nTop countries by year (number of presentations):")
    print(f"{'Country':<15} " + " ".join(f"{year:>7}" for year in years))
    print("-" * (15 + 8 * len(years)))
    
    for row in table_data:
        country = row[0]
        counts = row[1:]
        print(f"{country:<15} " + " ".join(f"{count:>7}" for count in counts))
    
    return unique_countries_by_year, hhi_by_year

def create_regional_diversity_plot(country_counts, conference_data):
    """
    Create plots showing regional diversity of participation.
    
    Parameters:
    - country_counts: Counter object with country counts across all conferences
    - conference_data: Dictionary with conference data by year
    """
    # Define regions and country mappings
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Netherlands', 
                  'Spain', 'Poland', 'Russia', 'Finland', 'Sweden', 'Norway', 'Denmark',
                  'Czech Republic', 'Hungary', 'Austria', 'Belgium', 'Portugal', 'Greece',
                  'Ireland', 'Romania', 'Bulgaria', 'Croatia', 'Serbia', 'Slovakia', 
                  'Slovenia', 'Ukraine', 'Estonia', 'Latvia', 'Lithuania'],
        'Asia': ['Japan', 'China', 'India', 'Korea', 'Taiwan', 'Singapore', 'Malaysia',
                'Thailand', 'Vietnam', 'Indonesia', 'Philippines', 'Pakistan', 'Bangladesh',
                'Israel', 'Turkey', 'Iran', 'Iraq', 'Saudi Arabia', 'UAE', 'Qatar'],
        'Other': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 
                 'Australia', 'New Zealand', 'South Africa', 'Egypt', 'Morocco', 'Tunisia',
                 'Nigeria', 'Kenya', 'Ethiopia', 'Ghana', 'Senegal']
    }
    
    # Function to map country to region
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return 'Other'
    
    # Calculate regional counts across all conferences
    region_counts = Counter()
    for country, count in country_counts.items():
        if country != 'Unknown':
            region = get_region(country)
            region_counts[region] += count
    
    # Create pie chart for overall regional distribution
    plt.figure(figsize=(12, 10))
    
    # Create a 1x2 subplot grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
    
    # First subplot: Pie chart
    ax1 = plt.subplot(gs[0])
    
    # Define colors for regions
    region_colors = {
        'North America': '#1f77b4',  # Blue
        'Europe': '#ff7f0e',         # Orange
        'Asia': '#2ca02c',           # Green
        'Other': '#d62728'           # Red
    }
    
    # Prepare data for pie chart
    labels = region_counts.keys()
    sizes = region_counts.values()
    colors = [region_colors[region] for region in labels]
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Make percentage labels more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax1.set_title('Overall Regional Distribution of Contributions')
    
    # Second subplot: Region trends over time
    ax2 = plt.subplot(gs[1])
    
    # Get sorted years
    years = sorted([year for year in conference_data.keys() if year.isdigit()])
    
    # Calculate regional percentages for each year
    region_percentages = {region: [] for region in regions.keys()}
    
    for year in years:
        data = conference_data[year]
        yearly_region_counts = Counter()
        
        # Get talk data
        all_talks = []
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                all_talks.extend(data[talk_type])
        
        # Count by region
        for talk in all_talks:
            country = talk.get('Country', 'Unknown')
            if country != 'Unknown':
                region = get_region(country)
                yearly_region_counts[region] += 1
        
        # Calculate percentages
        total = sum(yearly_region_counts.values())
        if total > 0:
            for region in regions.keys():
                percentage = (yearly_region_counts.get(region, 0) / total) * 100
                region_percentages[region].append(percentage)
        else:
            for region in regions.keys():
                region_percentages[region].append(0)
    
    # Plot trends
    for region, percentages in region_percentages.items():
        ax2.plot(years, percentages, 'o-', label=region, color=region_colors[region], linewidth=2)
    
    ax2.set_title('Regional Representation Over Time')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage of Contributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/regional_diversity.pdf', bbox_inches='tight')
    
    # Print regional statistics
    print("\nRegional Distribution of Contributions:")
    total = sum(region_counts.values())
    for region, count in region_counts.most_common():
        percentage = (count / total) * 100
        print(f"{region}: {count} ({percentage:.1f}%)")

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

def create_institute_bubble_chart(institute_counts, top_n=30):
    """
    Create a bubble chart visualization of the top institutes.
    
    Parameters:
    - institute_counts: Counter object with institute counts
    - top_n: Number of top institutes to include in the visualization
    """
    print(f"Creating institute bubble chart for top {top_n} institutes...")
    
    # Make sure output directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Get the top institutes
    top_institutes = institute_counts.most_common(top_n)
    
    # Extract institute names and counts
    institutes = [inst[0] for inst in top_institutes]
    counts = [inst[1] for inst in top_institutes]
    
    # Create a DataFrame for the visualization
    df = pd.DataFrame({
        'Institute': institutes,
        'Count': counts
    })
    
    # Calculate bubble sizes (sqrt-scaled for better visualization)
    df['BubbleSize'] = np.sqrt(df['Count']) * 5
    
    # Generate random x and y coordinates for initial positioning
    np.random.seed(42)  # For reproducibility
    df['x'] = np.random.rand(len(df)) * 100
    df['y'] = np.random.rand(len(df)) * 100
    
    # Create the figure
    plt.figure(figsize=(16, 12))
    
    # Create bubble chart
    plt.scatter(df['x'], df['y'], s=df['BubbleSize']**2, alpha=0.6, 
                edgecolors='black', linewidth=1)
    
    # Add labels to each bubble
    for i, row in df.iterrows():
        plt.text(row['x'], row['y'], f"{row['Institute']}\n({row['Count']})", 
                 fontsize=9, ha='center', va='center')
    
    # Remove axes and gridlines for cleaner look
    plt.axis('off')
    
    # Add title
    plt.title(f'Top {top_n} Institutes by Presentation Count', fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('figures/institute_bubble_chart.pdf', bbox_inches='tight')
    plt.savefig('figures/institute_bubble_chart.png', dpi=300, bbox_inches='tight')
    
    print(f"Institute bubble chart saved to figures/institute_bubble_chart.pdf")

def create_country_trends_plot(years, plenary_data, parallel_data, figures_dir):
    """
    Create a line chart showing trends in country representation over time.
    
    Parameters:
    - years: List of conference years
    - plenary_data: Dictionary mapping years to country counts for plenary talks
    - parallel_data: Dictionary mapping years to country counts for parallel talks
    - figures_dir: Directory to save the figures
    """
    print("\nCreating country trends plot with enhanced mapping...")
    
    # Get all countries that appear across all years
    all_countries = set()
    for year in years:
        all_countries.update(plenary_data.get(year, {}).keys())
        all_countries.update(parallel_data.get(year, {}).keys())
    
    # Remove 'Unknown' from the set of countries
    if 'Unknown' in all_countries:
        all_countries.remove('Unknown')
    
    # Calculate total contributions by country across all years
    country_total = Counter()
    for year in years:
        # Combine plenary and parallel data
        year_data = Counter(plenary_data.get(year, {}))
        year_data.update(parallel_data.get(year, {}))
        
        # Only count known countries
        for country, count in year_data.items():
            if country != 'Unknown' and country != '':
                country_total[country] += count
    
    # Select top countries by total contributions
    top_countries = [country for country, _ in country_total.most_common(8) 
                   if country != 'Unknown' and country != '']
    
    # Add emerging countries for trend analysis if they're in the data
    emerging_countries = ['Brazil', 'Poland', 'Czech Republic', 'South Africa', 'Finland']
    emerging_in_data = [c for c in emerging_countries if c in all_countries and c not in top_countries]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for top countries
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    # Plot the trends for top countries
    for i, country in enumerate(top_countries):
        country_percentages = [plenary_data[year].get(country, 0) + parallel_data[year].get(country, 0) 
                               for year in years]
        plt.plot(years, country_percentages, marker='o', linewidth=2, color=colors[i], label=country)
    
    # Plot the trends for emerging countries with dashed lines
    colors_emerging = plt.cm.Set2(np.linspace(0, 1, len(emerging_in_data)))
    for i, country in enumerate(emerging_in_data):
        country_percentages = [plenary_data[year].get(country, 0) + parallel_data[year].get(country, 0) 
                               for year in years]
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
    plt.savefig(os.path.join(figures_dir, 'country_trends_over_time.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Country trends chart saved to {os.path.join(figures_dir, 'country_trends_over_time.pdf')}")
    
    # Create a table with most common countries each year
    top_n = 10
    top_countries_all_time = [country for country, _ in country_total.most_common(top_n)]
    
    # Find the top countries across all years
    all_time_counts = Counter()
    for year_counts in country_total.values():
        all_time_counts.update(year_counts)
    
    top_countries_all_time = [country for country, _ in all_time_counts.most_common(top_n)]
    
    # Create a table with years in columns and countries in rows
    table_data = []
    for country in top_countries_all_time:
        row = [country]
        for year in years:
            row.append(country_total.get(country, 0))
        table_data.append(row)
    
    # Print the table
    print("\nTop countries by year (number of presentations):")
    print(f"{'Country':<15} " + " ".join(f"{year:>7}" for year in years))
    print("-" * (15 + 8 * len(years)))
    
    for row in table_data:
        country = row[0]
        counts = row[1:]
        print(f"{country:<15} " + " ".join(f"{count:>7}" for count in counts))
    
    return top_countries_all_time, country_total

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

def create_representation_ratio_plot(plenary_country, parallel_country):
    """
    Create a plot showing the representation ratio between plenary and parallel talks by country.
    
    Parameters:
    - plenary_country: Counter object with country counts for plenary talks
    - parallel_country: Counter object with country counts for parallel talks
    """
    # Calculate total counts
    plenary_total = sum(plenary_country.values())
    parallel_total = sum(parallel_country.values())
    
    # Get countries with at least 2 plenary talks
    countries = [country for country, count in plenary_country.items() if count >= 2]
    
    # Calculate representation ratios
    ratios = []
    for country in countries:
        plenary_share = plenary_country.get(country, 0) / plenary_total
        parallel_share = parallel_country.get(country, 0) / parallel_total
        
        # Avoid division by zero
        if parallel_share > 0:
            ratio = plenary_share / parallel_share
        else:
            ratio = 0
            
        ratios.append((country, ratio))
    
    # Sort by ratio
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Take top and bottom 10
    top_10 = ratios[:10]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_10))
    
    # Extract names and ratios
    names = [item[0] for item in top_10]
    values = [item[1] for item in top_10]
    
    # Create horizontal bars with color based on over/under representation
    colors = ['#2ca02c' if ratio > 1 else '#d62728' for ratio in values]
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add a vertical line at ratio=1 (equal representation)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    
    plt.yticks(y_pos, names)
    plt.xlabel('Representation Ratio (Plenary Share / Parallel Share)')
    plt.title('Representation Ratio between Plenary and Parallel Talks by Country')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/representation_ratio.pdf', bbox_inches='tight')
    
    # Print the ratios
    print("\nRepresentation Ratios (Plenary Share / Parallel Share):")
    for country, ratio in ratios:
        plenary_count = plenary_country.get(country, 0)
        parallel_count = parallel_country.get(country, 0)
        print(f"{country}: {ratio:.2f} ({plenary_count} plenary, {parallel_count} parallel)")

def create_unknown_institutes_report(all_conference_data, output_file='unknown_institutes_report.txt'):
    """
    Creates a detailed report about speakers with unknown institutes to help with data cleaning.
    
    Parameters:
    - all_conference_data: Dictionary with conference data
    - output_file: File to save the report
    """
    print("\nGenerating detailed report of unknown institutes...")
    
    unknown_by_year = {}
    unknown_speaker_details = {}
    unknown_institute_counts = {}
    
    # Extract unknown institutes and gather context
    for year, data in sorted(all_conference_data.items()):
        unknown_by_year[year] = []
        
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            institute = talk.get('Institute', '')
            country = talk.get('Country', '')
            
            if (not country or country == 'Unknown') and institute:
                # Record the unknown institute with context
                unknown_by_year[year].append({
                    'year': year,
                    'session_type': 'Plenary',
                    'institute': institute,
                    'speaker': talk.get('Speaker', 'Unknown'),
                    'title': talk.get('Title', 'Unknown')
                })
                
                # Count occurrences of each unknown institute
                unknown_institute_counts[institute] = unknown_institute_counts.get(institute, 0) + 1
                
                # Store details of speakers using this institute
                if institute not in unknown_speaker_details:
                    unknown_speaker_details[institute] = []
                
                unknown_speaker_details[institute].append({
                    'year': year,
                    'session_type': 'Plenary',
                    'speaker': talk.get('Speaker', 'Unknown'),
                    'title': talk.get('Title', 'Unknown')
                })
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            institute = talk.get('Institute', '')
            country = talk.get('Country', '')
            
            if (not country or country == 'Unknown') and institute:
                # Record the unknown institute with context
                unknown_by_year[year].append({
                    'year': year,
                    'session_type': 'Parallel',
                    'institute': institute,
                    'speaker': talk.get('Speaker', 'Unknown'),
                    'title': talk.get('Title', 'Unknown')
                })
                
                # Count occurrences of each unknown institute
                unknown_institute_counts[institute] = unknown_institute_counts.get(institute, 0) + 1
                
                # Store details of speakers using this institute
                if institute not in unknown_speaker_details:
                    unknown_speaker_details[institute] = []
                
                unknown_speaker_details[institute].append({
                    'year': year,
                    'session_type': 'Parallel',
                    'speaker': talk.get('Speaker', 'Unknown'),
                    'title': talk.get('Title', 'Unknown')
                })
    
    # Generate the report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# DETAILED REPORT OF UNKNOWN INSTITUTES\n")
        f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # Summary statistics
        total_unknown = sum(len(items) for items in unknown_by_year.values())
        unique_unknown = len(unknown_institute_counts)
        f.write(f"Total speakers with unknown country: {total_unknown}\n")
        f.write(f"Unique institutes without country mapping: {unique_unknown}\n\n")
        
        # Write most frequent unknowns
        f.write("## MOST FREQUENT UNKNOWN INSTITUTES\n")
        f.write("Format: Count, Institute, Speakers\n\n")
        
        for institute, count in sorted(unknown_institute_counts.items(), key=lambda x: x[1], reverse=True):
            speakers = set(item['speaker'] for item in unknown_speaker_details[institute])
            speakers_str = ", ".join(speakers)
            f.write(f"{count}, {institute}, [{speakers_str}]\n")
        
        f.write("\n\n## SUGGESTED MAPPINGS\n")
        f.write("Format: Institute,Country\n\n")
        
        # Attempt to suggest mappings for common patterns
        country_patterns = {
            'USA': ['University of', 'State University', 'College', 'Laboratory', 'National Lab', 'MIT'],
            'Germany': ['Universität', 'University of', 'Institut für', 'GSI', 'Heidelberg', 'München', 'Berlin', 'Darmstadt'],
            'France': ['Université', 'Institut', 'École', 'CNRS', 'CEA', 'Saclay', 'Nantes', 'Strasbourg'],
            'UK': ['University of', 'College London', 'Cambridge', 'Oxford', 'Birmingham', 'Manchester'],
            'Japan': ['University of Tokyo', 'Kyoto', 'Osaka', 'Tsukuba', 'RIKEN'],
            'China': ['Tsinghua', 'Peking', 'Fudan', 'USTC', 'CAS', 'Beijing', 'Shanghai'],
            'Italy': ['Università', 'INFN', 'Politecnico', 'Padova', 'Bologna', 'Torino', 'Roma', 'Milano'],
            'India': ['Institute of', 'University', 'IIT', 'Bhubaneswar', 'Kolkata', 'Mumbai', 'Delhi'],
            'Russia': ['Moscow', 'Petersburg', 'JINR', 'Kurchatov', 'Novosibirsk', 'Lebedev']
        }
        
        for institute in sorted(unknown_institute_counts.keys()):
            # Try to suggest a country based on patterns
            suggested_country = None
            confidence = 0
            
            for country, patterns in country_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in institute.lower():
                        # Calculate a simple confidence score based on pattern length
                        pattern_confidence = len(pattern) / len(institute)
                        if pattern_confidence > confidence:
                            suggested_country = country
                            confidence = pattern_confidence
            
            # Write the suggestion if confidence is above threshold
            if suggested_country and confidence > 0.2:
                f.write(f"{institute},{suggested_country}  # Confidence: {confidence:.2f}\n")
            else:
                f.write(f"{institute},\n")
        
        f.write("\n\n## DETAILS BY CONFERENCE YEAR\n\n")
        
        # Write details by year
        for year, items in sorted(unknown_by_year.items()):
            if items:
                f.write(f"### Year: {year} ({len(items)} unknown institutes)\n\n")
                for item in items:
                    f.write(f"- {item['session_type']}: {item['speaker']} ({item['institute']})\n")
                    f.write(f"  Title: {item['title']}\n\n")
        
    print(f"Detailed unknown institutes report saved to {output_file}")
    
    # Also save a simplified CSV mapping file for easy editing
    csv_file = 'unknown_institute_mappings.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Institute,Country\n")
        for institute in sorted(unknown_institute_counts.keys()):
            # Try to suggest a country based on patterns
            suggested_country = None
            confidence = 0
            
            for country, patterns in country_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in institute.lower():
                        pattern_confidence = len(pattern) / len(institute)
                        if pattern_confidence > confidence:
                            suggested_country = country
                            confidence = pattern_confidence
            
            if suggested_country and confidence > 0.3:
                f.write(f"{institute},{suggested_country}\n")
            else:
                f.write(f"{institute},\n")
    
    print(f"CSV mapping template saved to {csv_file}")

def extract_affiliation_info(talk_data):
    """
    Extract institute and country information more effectively from talk data.
    Handles various formatting patterns and edge cases.
    
    Parameters:
    - talk_data: Dictionary containing talk information
    
    Returns:
    - Tuple of (institute, country)
    """
    institute = talk_data.get('Institute', '').strip()
    country = talk_data.get('Country', '').strip()
    speaker = talk_data.get('Speaker', '').strip()
    
    # If both fields are filled, no extraction needed
    if institute and institute.lower() != 'unknown' and country and country.lower() != 'unknown':
        return institute, country
    
    # Pattern 1: "Speaker Name (Institution)"
    institution_in_speaker = re.search(r'\(([^()]*)\)$', speaker)
    if institution_in_speaker:
        potential_inst = institution_in_speaker.group(1).strip()
        # Don't use it if it's just a country code
        if not re.match(r'^[A-Z]{2}$', potential_inst) and len(potential_inst) > 2:
            institute = potential_inst if not institute or institute.lower() == 'unknown' else institute
    
    # Pattern 2: "Institution (XX)" - extract country from code in parentheses
    country_code_match = re.search(r'\(([A-Z]{2})\)$', institute)
    if country_code_match and (not country or country.lower() == 'unknown'):
        code = country_code_match.group(1)
        country_map = {
            'US': 'USA', 'UK': 'United Kingdom', 'DE': 'Germany', 'FR': 'France',
            'IT': 'Italy', 'JP': 'Japan', 'CN': 'China', 'KR': 'Korea',
            'CH': 'Switzerland', 'IN': 'India', 'BR': 'Brazil', 'CA': 'Canada',
            'AU': 'Australia', 'NL': 'Netherlands', 'ES': 'Spain', 'SE': 'Sweden',
            'DK': 'Denmark', 'NO': 'Norway', 'FI': 'Finland', 'PL': 'Poland',
            'CZ': 'Czech Republic', 'AT': 'Austria', 'BE': 'Belgium', 'PT': 'Portugal',
            # Add more country codes as needed
        }
        country = country_map.get(code, code)
        
        # Clean up the institute name by removing the country code if needed
        institute = re.sub(r'\s*\([A-Z]{2}\)$', '', institute).strip()
    
    # Pattern 3: Check for common affiliations in the title
    title = talk_data.get('Title', '').strip()
    experiment_affiliations = {
        'ALICE': {'institute': 'CERN', 'country': 'Switzerland'},
        'ATLAS': {'institute': 'CERN', 'country': 'Switzerland'},
        'CMS': {'institute': 'CERN', 'country': 'Switzerland'},
        'LHCb': {'institute': 'CERN', 'country': 'Switzerland'},
        'PHENIX': {'institute': 'Brookhaven National Laboratory', 'country': 'USA'},
        'STAR': {'institute': 'Brookhaven National Laboratory', 'country': 'USA'},
    }
    
    # Check if title mentions an experiment
    if (not institute or institute.lower() == 'unknown') and title:
        for exp, aff in experiment_affiliations.items():
            if f"Experiment overview: {exp}" in title or f"{exp} experiment" in title:
                institute = aff['institute'] if not institute or institute.lower() == 'unknown' else institute
                country = aff['country'] if not country or country.lower() == 'unknown' else country
                break
    
    # Pattern 4: Handle special cases for international organizations
    if institute in ['CERN', 'JINR', 'ESA', 'ITER'] and (not country or country.lower() == 'unknown'):
        special_locations = {
            'CERN': 'Switzerland', 'JINR': 'Russia', 'ESA': 'France', 'ITER': 'France'
        }
        country = special_locations.get(institute, country)
    
    # Pattern 5: Try to extract from format "LastName, FirstName (Institute (Country))"
    nested_parens = re.search(r'\((.*?)\s*\(([^()]+)\)\s*\)', speaker)
    if nested_parens:
        potential_inst = nested_parens.group(1).strip()
        potential_country = nested_parens.group(2).strip()
        
        if not institute or institute.lower() == 'unknown':
            institute = potential_inst
        
        if not country or country.lower() == 'unknown':
            # Check if it's a country name or code
            if potential_country in ["USA", "France", "Germany", "UK", "Japan", "China", "Italy"]:
                country = potential_country
            elif re.match(r'^[A-Z]{2}$', potential_country):
                country_map = {
                    'US': 'USA', 'UK': 'United Kingdom', 'DE': 'Germany', 'FR': 'France',
                    # (same map as above)
                }
                country = country_map.get(potential_country, potential_country)
    
    return institute, country

def load_participant_affiliations(registration_files=None, conference_indico_ids=None):
    """
    Load participant/registration data to extract speaker affiliations.
    
    Parameters:
    - registration_files: List of registration data files (CSV/JSON/Excel) or None for default filenames
    - conference_indico_ids: Dictionary mapping years to Indico event IDs to extract participant data (optional)
    
    Returns:
    - Dictionary mapping speaker names to their affiliations
    """
    if registration_files is None:
        # Default files to look for
        registration_files = [
            'data/registrations.csv',
            'data/participants.csv',
            'data/registrations.xlsx',
            'data/participants.xlsx',
            'data/registrations.json',
            'data/participants.json'
        ]
    
    participant_affiliations = {}
    loaded_count = 0
    
    print("\nAttempting to load participant/registration data...")
    
    # First, try to extract from Indico API if IDs are provided
    if conference_indico_ids and isinstance(conference_indico_ids, dict):
        for year, indico_id in conference_indico_ids.items():
            year_file = f'data/participants_{year}.csv'
            year_participants = extract_participants_from_contributions(indico_id, year, year_file)
            
            # Add to the combined dictionary
            participant_affiliations.update(year_participants)
            loaded_count += len(year_participants)
    
    # If no participants were loaded from API, try the files
    if loaded_count == 0:
        # Try to load each potential registration file
        for file_path in registration_files:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    # Load CSV file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Try different possible column names
                            name = (row.get('Name') or row.get('Participant') or 
                                   row.get('Full Name') or row.get('Attendee'))
                            institute = (row.get('Institute') or row.get('Institution') or 
                                        row.get('Affiliation') or row.get('Organization'))
                            country = (row.get('Country') or row.get('Nation') or 
                                      row.get('Country/Region'))
                            
                            if name and (institute or country):
                                # Normalize name format (LastName, FirstName)
                                if ',' not in name and ' ' in name:
                                    parts = name.strip().split()
                                    if len(parts) > 1:
                                        name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                                
                                participant_affiliations[name.strip()] = {
                                    'Institute': institute.strip() if institute else '',
                                    'Country': country.strip() if country else ''
                                }
                                loaded_count += 1
                
                elif file_ext == '.xlsx':
                    # Load Excel file (requires pandas)
                    try:
                        import pandas as pd
                        df = pd.read_excel(file_path)
                        
                        # Try different possible column names
                        name_col = next((col for col in df.columns if col.lower() in 
                                        ['name', 'participant', 'full name', 'attendee']), None)
                        inst_col = next((col for col in df.columns if col.lower() in 
                                        ['institute', 'institution', 'affiliation', 'organization']), None)
                        country_col = next((col for col in df.columns if col.lower() in 
                                           ['country', 'nation', 'country/region']), None)
                        
                        if name_col:
                            for _, row in df.iterrows():
                                name = str(row[name_col])
                                institute = str(row[inst_col]) if inst_col and not pd.isna(row[inst_col]) else ''
                                country = str(row[country_col]) if country_col and not pd.isna(row[country_col]) else ''
                                
                                if name and name != 'nan' and (institute or country):
                                    # Normalize name format
                                    if ',' not in name and ' ' in name:
                                        parts = name.strip().split()
                                        if len(parts) > 1:
                                            name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                                
                                    participant_affiliations[name.strip()] = {
                                        'Institute': institute.strip() if institute else '',
                                        'Country': country.strip() if country else ''
                                    }
                                    loaded_count += 1
                    
                    except ImportError:
                        print(f"Warning: pandas is required to read Excel files. Skipping {file_path}")
                
                elif file_ext == '.json':
                    # Load JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle different possible JSON structures
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    name = (item.get('Name') or item.get('Participant') or 
                                           item.get('FullName') or item.get('Attendee'))
                                    institute = (item.get('Institute') or item.get('Institution') or 
                                                item.get('Affiliation') or item.get('Organization'))
                                    country = (item.get('Country') or item.get('Nation') or 
                                              item.get('CountryRegion'))
                                    
                                    if name and (institute or country):
                                        # Normalize name format
                                        if ',' not in name and ' ' in name:
                                            parts = name.strip().split()
                                            if len(parts) > 1:
                                                name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                                        
                                        participant_affiliations[name.strip()] = {
                                            'Institute': institute.strip() if institute else '',
                                            'Country': country.strip() if country else ''
                                        }
                                        loaded_count += 1
                
                print(f"Loaded {loaded_count} participant affiliations from {file_path}")
                break  # Stop after successfully loading a file
                
            except Exception as e:
                # Just try the next file if one fails
                print(f"Could not load registrations from {file_path}: {str(e)}")
    
    # If we tried everything and found nothing, suggest creating a file
    if loaded_count == 0:
        print("\nNo registration/participant data found. You can:")
        print("1. Provide Indico IDs for conferences to extract participant data")
        print("2. Create a CSV file manually with this format:")
        print("   Name,Institute,Country")
        print("   LastName, FirstName,University of Example,Country")
        print("   Save as 'data/participants.csv'")

    # Create alternative name formats for better matching
    alt_names = {}
    for name, info in participant_affiliations.items():
        # Add variations for better matching
        
        # Create FirstName LastName format if we have LastName, FirstName
        if ',' in name:
            last_name, first_name = name.split(',', 1)
            alternative_format = f"{first_name.strip()} {last_name.strip()}"
            alt_names[alternative_format] = info
        
        # Create initial-based variations
        if ' ' in name:
            words = name.replace(',', '').split()
            if len(words) > 1:
                # First initial + last name
                first_initial = words[0][0] if words[0] else ''
                last_name = words[-1]
                alt_names[f"{first_initial}. {last_name}"] = info
                alt_names[f"{first_initial}.{last_name}"] = info
    
    # Add the alternative names to the main dictionary
    participant_affiliations.update(alt_names)
    
    print(f"Total unique participants (including name variations): {len(participant_affiliations)}")
    return participant_affiliations

def check_missing_speaker_affiliations(all_conference_data, output_file='missing_affiliations_report.txt', indico_ids_file='listofQMindigo'):
    """
    Identifies and reports speakers who are missing both institute and country information.
    
    Parameters:
    - all_conference_data: Dictionary with conference data
    - output_file: File to save the report
    - indico_ids_file: Path to file containing Indico IDs (default: listofQMindigo)
    """
    print("\nChecking for speakers missing both institute and country information...")
    
    # Load Indico IDs from file
    indico_ids = load_indico_ids_from_file(indico_ids_file)
    
    # First, try to load affiliation data from registration/participant lists
    participant_affiliations = load_participant_affiliations(conference_indico_ids=indico_ids)
    
    reg_matches_count = 0
    
    # Then, enhance the data by extracting affiliations more effectively
    print("Applying enhanced affiliation extraction...")
    extracted_count = 0
    
    for year, data in all_conference_data.items():
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            original_institute = talk.get('Institute', '')
            original_country = talk.get('Country', '')
            speaker = talk.get('Speaker', '').strip()
            
            # First try to match with registration data
            if speaker and (not original_institute or original_institute.lower() == 'unknown' or 
                           not original_country or original_country.lower() == 'unknown'):
                if speaker in participant_affiliations:
                    affiliation = participant_affiliations[speaker]
                    
                    if (not original_institute or original_institute.lower() == 'unknown') and affiliation.get('Institute'):
                        talk['Institute'] = affiliation['Institute']
                        reg_matches_count += 1
                    
                    if (not original_country or original_country.lower() == 'unknown') and affiliation.get('Country'):
                        talk['Country'] = affiliation['Country']
                        reg_matches_count += 1
            
            # Then apply enhanced extraction
            extracted_institute, extracted_country = extract_affiliation_info(talk)
            
            # Update the data if we've found better information
            if extracted_institute and (not talk.get('Institute') or talk.get('Institute', '').lower() == 'unknown'):
                talk['Institute'] = extracted_institute
                extracted_count += 1
            
            if extracted_country and (not talk.get('Country') or talk.get('Country', '').lower() == 'unknown'):
                talk['Country'] = extracted_country
                extracted_count += 1
        
        # Process parallel talks (same logic)
        for talk in data.get('parallel_talks', []):
            original_institute = talk.get('Institute', '')
            original_country = talk.get('Country', '')
            speaker = talk.get('Speaker', '').strip()
            
            # First try to match with registration data
            if speaker and (not original_institute or original_institute.lower() == 'unknown' or 
                           not original_country or original_country.lower() == 'unknown'):
                if speaker in participant_affiliations:
                    affiliation = participant_affiliations[speaker]
                    
                    if (not original_institute or original_institute.lower() == 'unknown') and affiliation.get('Institute'):
                        talk['Institute'] = affiliation['Institute']
                        reg_matches_count += 1
                    
                    if (not original_country or original_country.lower() == 'unknown') and affiliation.get('Country'):
                        talk['Country'] = affiliation['Country']
                        reg_matches_count += 1
            
            # Then apply enhanced extraction
            extracted_institute, extracted_country = extract_affiliation_info(talk)
            
            # Update the data if we've found better information
            if extracted_institute and (not talk.get('Institute') or talk.get('Institute', '').lower() == 'unknown'):
                talk['Institute'] = extracted_institute
                extracted_count += 1
            
            if extracted_country and (not talk.get('Country') or talk.get('Country', '').lower() == 'unknown'):
                talk['Country'] = extracted_country
                extracted_count += 1
    
    print(f"Registration data provided {reg_matches_count} missing institute/country values")
    print(f"Enhanced extraction found {extracted_count} additional missing institute/country values")
    
    # Now perform the regular check for remaining missing affiliations
    missing_by_year = {}
    missing_speaker_list = []
    speaker_talk_count = {}
    
    total_talks = 0
    missing_affiliations_count = 0
    
    # Extract speakers with missing information
    for year, data in sorted(all_conference_data.items()):
        missing_by_year[year] = []
        
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            total_talks += 1
            institute = talk.get('Institute', '').strip()
            country = talk.get('Country', '').strip()
            speaker = talk.get('Speaker', '').strip()
            
            if speaker and (not institute or institute.lower() == 'unknown') and (not country or country.lower() == 'unknown'):
                missing_affiliations_count += 1
                
                # Record the missing affiliation with context
                talk_info = {
                    'year': year,
                    'session_type': 'Plenary',
                    'speaker': speaker,
                    'title': talk.get('Title', 'Unknown'),
                    'track': talk.get('Track', 'Unknown')
                }
                
                missing_by_year[year].append(talk_info)
                missing_speaker_list.append(talk_info)
                
                # Count talks per speaker
                speaker_talk_count[speaker] = speaker_talk_count.get(speaker, 0) + 1
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            total_talks += 1
            institute = talk.get('Institute', '').strip()
            country = talk.get('Country', '').strip()
            speaker = talk.get('Speaker', '').strip()
            
            if speaker and (not institute or institute.lower() == 'unknown') and (not country or country.lower() == 'unknown'):
                missing_affiliations_count += 1
                
                # Record the missing affiliation with context
                talk_info = {
                    'year': year,
                    'session_type': 'Parallel',
                    'speaker': speaker,
                    'title': talk.get('Title', 'Unknown'),
                    'track': talk.get('Track', 'Unknown')
                }
                
                missing_by_year[year].append(talk_info)
                missing_speaker_list.append(talk_info)
                
                # Count talks per speaker
                speaker_talk_count[speaker] = speaker_talk_count.get(speaker, 0) + 1
    
    # Generate the report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SPEAKERS WITH MISSING AFFILIATIONS (NO INSTITUTE AND NO COUNTRY)\n")
        f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # Summary statistics
        f.write(f"Total talks analyzed: {total_talks}\n")
        f.write(f"Registration data provided {reg_matches_count} missing institute/country values\n")
        f.write(f"Enhanced extraction found {extracted_count} missing institute/country values\n")
        f.write(f"Talks with speakers still missing both institute and country: {missing_affiliations_count} ({missing_affiliations_count/total_talks*100:.2f}%)\n")
        f.write(f"Unique speakers missing affiliations: {len(speaker_talk_count)}\n\n")
        
        # Write most frequent speakers with missing affiliations
        f.write("## SPEAKERS WITH MULTIPLE TALKS MISSING AFFILIATIONS\n")
        f.write("These speakers should be prioritized for data cleanup as they appear multiple times:\n\n")
        
        for speaker, count in sorted(speaker_talk_count.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                f.write(f"{speaker}: {count} talks\n")
        
        f.write("\n\n## DETAILS BY CONFERENCE YEAR\n\n")
        
        # Write details by year
        for year, items in sorted(missing_by_year.items()):
            if items:
                f.write(f"### Year: {year} ({len(items)} speakers with missing affiliations)\n\n")
                for item in items:
                    f.write(f"- {item['session_type']}: {item['speaker']}\n")
                    f.write(f"  Title: {item['title']}\n")
                    if 'track' in item and item['track'] and item['track'] != 'Unknown':
                        f.write(f"  Track: {item['track']}\n")
                    f.write("\n")
        
        # Provide guidance on resolving missing information
        f.write("\n## HOW TO RESOLVE MISSING AFFILIATIONS\n\n")
        f.write("1. Check the conference website or program for speaker information\n")
        f.write("2. Search for the speaker online using the talk title as context\n")
        f.write("3. Look for other talks by the same speaker in different years\n")
        f.write("4. Consider reaching out to conference organizers for historical records\n\n")
        
        # Create a template for filling in the missing data
        f.write("## TEMPLATE FOR COMPLETING MISSING DATA\n")
        f.write("Copy, fill in, and use to update your database:\n\n")
        
        f.write("```\n")
        for speaker in sorted(speaker_talk_count.keys()):
            f.write(f"{speaker},Institute,Country\n")
        f.write("```\n")
    
    print(f"Missing affiliations report saved to {output_file}")
    
    # Also create a CSV file for easier data entry
    csv_file = 'missing_speaker_affiliations.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Speaker,Institute,Country,Year,Session_Type,Title\n")
        for item in sorted(missing_speaker_list, key=lambda x: x['speaker']):
            speaker = item['speaker'].replace('"', '""')  # Escape quotes for CSV
            title = item['title'].replace('"', '""')
            f.write(f'"{speaker}",,,"{ item["year"]}","{item["session_type"]}","{title}"\n')
    
    print(f"CSV template for missing affiliations saved to {csv_file}")
    
    return missing_affiliations_count > 0

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

def extract_participant_counts(conference_indico_ids):
    """
    Extract the total number of unique participants for each conference year.
    
    Parameters:
    - conference_indico_ids: Dictionary mapping years to Indico event IDs
    
    Returns:
    - Dictionary mapping years to participant counts
    """
    participant_counts = {}
    
    print("\nExtracting participant counts for each conference year...")
    
    for year, indico_id in conference_indico_ids.items():
        try:
            url = f"https://indico.cern.ch/export/event/{indico_id}.json?detail=contributions&pretty=yes"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print(f"No data found for year {year}")
                continue
                
            event_data = results[0]
            contributions = event_data.get('contributions', [])
            
            # Use a set to count unique participants
            unique_participants = set()
            
            for contribution in contributions:
                # Extract all person types
                all_persons = []
                
                # Speakers
                speakers = contribution.get('speakers', [])
                if speakers:
                    all_persons.extend(speakers)
                
                # Person links
                person_links = contribution.get('person_links', [])
                if person_links:
                    all_persons.extend(person_links)
                
                # Primary authors
                primary_authors = contribution.get('primary_authors', [])
                if primary_authors:
                    all_persons.extend(primary_authors)
                
                # Coauthors
                coauthors = contribution.get('coauthors', [])
                if coauthors:
                    all_persons.extend(coauthors)
                
                # Extract unique IDs or names
                for person in all_persons:
                    person_id = person.get('id', None)
                    if person_id:
                        unique_participants.add(person_id)
                    else:
                        # If no ID, use name as fallback
                        name = (person.get('fullName') or 
                                f"{person.get('first_name', '')} {person.get('last_name', '')}" or 
                                person.get('name', '')).strip()
                        if name:
                            unique_participants.add(name)
            
            participant_counts[year] = len(unique_participants)
            print(f"Year {year}: {participant_counts[year]} unique participants")
            
        except Exception as e:
            print(f"Error extracting participant count for year {year}: {str(e)}")
    
    return participant_counts

# Update the visualization function to include participant counts
def create_summary_statistics_plot(conference_data, filename="figs/QM_talk_statistics.pdf"):
    """
    Create a comprehensive plot of QM conference statistics with participant counts.
    """
    print("\nCreating summary statistics plot...")
    
    # Load Indico IDs from file to get participant counts
    indico_ids = load_indico_ids_from_file('listofQMindigo')
    participant_counts = extract_participant_counts(indico_ids)
    
    # Set up the plot
    fig = plt.figure(figsize=(14, 10))
    years = sorted([int(year) for year in conference_data.keys() if year.isdigit()])
    x = np.array(years)
    
    # Create GridSpec for layout
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 1])
    
    # Plot 1: Talk Counts by Type
    ax1 = plt.subplot(gs[0])
    plenary_counts = []
    parallel_counts = []
    poster_counts = []
    flash_counts = []
    participant_data = []
    
    for year in years:
        year_str = str(year)
        if year_str in conference_data:
            plenary_counts.append(len(conference_data[year_str].get('plenary_talks', [])))
            parallel_counts.append(len(conference_data[year_str].get('parallel_talks', [])))
            poster_counts.append(len(conference_data[year_str].get('poster_talks', [])))
            flash_counts.append(FLASH_TALK_COUNTS.get(year_str, 0))
            
            # Add participant count if available, or 0 if not
            if year_str in participant_counts:
                participant_data.append(participant_counts[year_str])
            else:
                participant_data.append(0)
    
    # Create the stacked bar chart for talk counts
    width = 0.8
    bars1 = ax1.bar(x, plenary_counts, width, label='Plenary Talks', color='#1f77b4')
    bars2 = ax1.bar(x, parallel_counts, width, label='Parallel Talks', bottom=plenary_counts, color='#ff7f0e')
    
    # Calculate cumulative heights for stacking
    bottom = np.array(plenary_counts) + np.array(parallel_counts)
    bars3 = ax1.bar(x, poster_counts, width, label='Posters', bottom=bottom, color='#2ca02c')
    
    # Add Flash Talks
    bottom = bottom + np.array(poster_counts)
    bars4 = ax1.bar(x, flash_counts, width, label='Flash Talks', bottom=bottom, color='#d62728')
    
    # Add participant count line on secondary y-axis
    ax1_2 = ax1.twinx()
    participant_line = ax1_2.plot(x, participant_data, 'o-', color='purple', linewidth=2, label='Participants')
    ax1_2.set_ylabel('Number of Participants', color='purple')
    ax1_2.tick_params(axis='y', labelcolor='purple')
    
    # Add annotations for participant counts
    for i, count in enumerate(participant_data):
        if count > 0:  # Only annotate if we have data
            ax1_2.annotate(f"{count}", 
                         (x[i], count),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center',
                         fontsize=9,
                         color='purple')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel('Number of Talks')
    ax1.set_title('QM Conference Statistics by Year', fontsize=16)
    
    # Add annotations for total talk counts
    for i in range(len(x)):
        total = plenary_counts[i] + parallel_counts[i] + poster_counts[i] + flash_counts[i]
        ax1.text(x[i], total + 5, f"{total}", ha='center', va='bottom')
    
    # Rest of the visualization code remains the same
    # ...

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

def create_plenary_country_plot(plenary_country):
    """
    Create a horizontal bar chart showing the distribution of plenary talks by country.
    
    Parameters:
    - plenary_country: Counter object with country counts for plenary talks
    """
    # Get the top countries
    top_countries = plenary_country.most_common(15)  # Show top 15 countries
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_countries))
    
    # Calculate total for percentages
    total = sum(plenary_country.values())
    
    # Extract names and counts
    names = [item[0] for item in top_countries]
    values = [item[1] for item in top_countries]
    
    # Define region-based colors for visual grouping
    # Define basic color map for regions
    region_colors = {
        'North America': '#1f77b4',  # Blue
        'Europe': '#ff7f0e',         # Orange
        'Asia': '#2ca02c',           # Green
        'Other': '#d62728'           # Red
    }
    
    # Map countries to regions
    country_regions = {
        'USA': 'North America',
        'Germany': 'Europe',
        'France': 'Europe', 
        'UK': 'Europe',
        'Italy': 'Europe',
        'Switzerland': 'Europe',
        'Netherlands': 'Europe',
        'Spain': 'Europe',
        'Poland': 'Europe',
        'Finland': 'Europe',
        'Japan': 'Asia',
        'China': 'Asia',
        'India': 'Asia',
        'Korea': 'Asia',
        'Canada': 'North America',
        'Brazil': 'Other',
        'Australia': 'Other',
        'Russia': 'Europe',
        'South Africa': 'Other',
        'Mexico': 'North America'
    }
    
    # Assign colors based on regions
    colors = [region_colors.get(country_regions.get(country, 'Other'), '#7f7f7f') for country in names]
    
    # Create horizontal bars
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add percentages to the end of each bar
    for i, (country, count) in enumerate(top_countries):
        percentage = (count / total) * 100
        plt.text(count + 0.5, i, f"{percentage:.1f}%", va='center')
    
    plt.yticks(y_pos, names)
    plt.xlabel('Number of Plenary Talks')
    plt.title('Distribution of Plenary Talks by Country')
    plt.tight_layout()
    
    # Create legend for regions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=region)
                      for region, color in region_colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Save the figure
    plt.savefig('figures/plenary_talks_by_country.pdf', bbox_inches='tight')
    
    # Print the top countries
    print("\nTop Countries by Plenary Talks:")
    for i, (country, count) in enumerate(top_countries, 1):
        percentage = (count / total) * 100
        print(f"{i}. {country}: {count} ({percentage:.1f}%)")

def create_representation_ratio_plot(plenary_country, parallel_country):
    """
    Create a plot showing the representation ratio between plenary and parallel talks by country.
    
    Parameters:
    - plenary_country: Counter object with country counts for plenary talks
    - parallel_country: Counter object with country counts for parallel talks
    """
    # Calculate total counts
    plenary_total = sum(plenary_country.values())
    parallel_total = sum(parallel_country.values())
    
    # Get countries with at least 2 plenary talks
    countries = [country for country, count in plenary_country.items() if count >= 2]
    
    # Calculate representation ratios
    ratios = []
    for country in countries:
        plenary_share = plenary_country.get(country, 0) / plenary_total
        parallel_share = parallel_country.get(country, 0) / parallel_total
        
        # Avoid division by zero
        if parallel_share > 0:
            ratio = plenary_share / parallel_share
        else:
            ratio = 0
            
        ratios.append((country, ratio))
    
    # Sort by ratio
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Take top and bottom 10
    top_10 = ratios[:10]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_10))
    
    # Extract names and ratios
    names = [item[0] for item in top_10]
    values = [item[1] for item in top_10]
    
    # Create horizontal bars with color based on over/under representation
    colors = ['#2ca02c' if ratio > 1 else '#d62728' for ratio in values]
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add a vertical line at ratio=1 (equal representation)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    
    plt.yticks(y_pos, names)
    plt.xlabel('Representation Ratio (Plenary Share / Parallel Share)')
    plt.title('Representation Ratio between Plenary and Parallel Talks by Country')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/representation_ratio.pdf', bbox_inches='tight')
    
    # Print the ratios
    print("\nRepresentation Ratios (Plenary Share / Parallel Share):")
    for country, ratio in ratios:
        plenary_count = plenary_country.get(country, 0)
        parallel_count = parallel_country.get(country, 0)
        print(f"{country}: {ratio:.2f} ({plenary_count} plenary, {parallel_count} parallel)")

def create_regional_diversity_plot(country_counts, conference_data):
    """
    Create plots showing regional diversity of participation.
    
    Parameters:
    - country_counts: Counter object with country counts across all conferences
    - conference_data: Dictionary with conference data by year
    """
    # Define regions and country mappings
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Netherlands', 
                  'Spain', 'Poland', 'Russia', 'Finland', 'Sweden', 'Norway', 'Denmark',
                  'Czech Republic', 'Hungary', 'Austria', 'Belgium', 'Portugal', 'Greece',
                  'Ireland', 'Romania', 'Bulgaria', 'Croatia', 'Serbia', 'Slovakia', 
                  'Slovenia', 'Ukraine', 'Estonia', 'Latvia', 'Lithuania'],
        'Asia': ['Japan', 'China', 'India', 'Korea', 'Taiwan', 'Singapore', 'Malaysia',
                'Thailand', 'Vietnam', 'Indonesia', 'Philippines', 'Pakistan', 'Bangladesh',
                'Israel', 'Turkey', 'Iran', 'Iraq', 'Saudi Arabia', 'UAE', 'Qatar'],
        'Other': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 
                 'Australia', 'New Zealand', 'South Africa', 'Egypt', 'Morocco', 'Tunisia',
                 'Nigeria', 'Kenya', 'Ethiopia', 'Ghana', 'Senegal']
    }
    
    # Function to map country to region
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return 'Other'
    
    # Calculate regional counts across all conferences
    region_counts = Counter()
    for country, count in country_counts.items():
        if country != 'Unknown':
            region = get_region(country)
            region_counts[region] += count
    
    # Create pie chart for overall regional distribution
    plt.figure(figsize=(12, 10))
    
    # Create a 1x2 subplot grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
    
    # First subplot: Pie chart
    ax1 = plt.subplot(gs[0])
    
    # Define colors for regions
    region_colors = {
        'North America': '#1f77b4',  # Blue
        'Europe': '#ff7f0e',         # Orange
        'Asia': '#2ca02c',           # Green
        'Other': '#d62728'           # Red
    }
    
    # Prepare data for pie chart
    labels = region_counts.keys()
    sizes = region_counts.values()
    colors = [region_colors[region] for region in labels]
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Make percentage labels more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax1.set_title('Overall Regional Distribution of Contributions')
    
    # Second subplot: Region trends over time
    ax2 = plt.subplot(gs[1])
    
    # Get sorted years
    years = sorted([year for year in conference_data.keys() if year.isdigit()])
    
    # Calculate regional percentages for each year
    region_percentages = {region: [] for region in regions.keys()}
    
    for year in years:
        data = conference_data[year]
        yearly_region_counts = Counter()
        
        # Get talk data
        all_talks = []
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                all_talks.extend(data[talk_type])
        
        # Count by region
        for talk in all_talks:
            country = talk.get('Country', 'Unknown')
            if country != 'Unknown':
                region = get_region(country)
                yearly_region_counts[region] += 1
        
        # Calculate percentages
        total = sum(yearly_region_counts.values())
        if total > 0:
            for region in regions.keys():
                percentage = (yearly_region_counts.get(region, 0) / total) * 100
                region_percentages[region].append(percentage)
        else:
            for region in regions.keys():
                region_percentages[region].append(0)
    
    # Plot trends
    for region, percentages in region_percentages.items():
        ax2.plot(years, percentages, 'o-', label=region, color=region_colors[region], linewidth=2)
    
    ax2.set_title('Regional Representation Over Time')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage of Contributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/regional_diversity.pdf', bbox_inches='tight')
    
    # Print regional statistics
    print("\nRegional Distribution of Contributions:")
    total = sum(region_counts.values())
    for region, count in region_counts.most_common():
        percentage = (count / total) * 100
        print(f"{region}: {count} ({percentage:.1f}%)")

# Add an updated version that handles the 4-argument call
def create_regional_diversity_plot_by_year(years, plenary_data, parallel_data, figures_dir):
    """
    Create plots showing regional diversity of participation by year.
    
    Parameters:
    - years: List of conference years
    - plenary_data: Dictionary with plenary talk data by country
    - parallel_data: Dictionary with parallel talk data by country
    - figures_dir: Directory to save figures
    """
    print("\nCreating regional diversity plot by year...")
    
    # Ensure the figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # Define regions and country mappings
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Netherlands', 
                  'Spain', 'Poland', 'Russia', 'Finland', 'Sweden', 'Norway', 'Denmark',
                  'Czech Republic', 'Hungary', 'Austria', 'Belgium', 'Portugal', 'Greece',
                  'Ireland', 'Romania', 'Bulgaria', 'Croatia', 'Serbia', 'Slovakia', 
                  'Slovenia', 'Ukraine', 'Estonia', 'Latvia', 'Lithuania'],
        'Asia': ['Japan', 'China', 'India', 'Korea', 'Taiwan', 'Singapore', 'Malaysia',
                'Thailand', 'Vietnam', 'Indonesia', 'Philippines', 'Pakistan', 'Bangladesh',
                'Israel', 'Turkey', 'Iran', 'Iraq', 'Saudi Arabia', 'UAE', 'Qatar'],
        'Other': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 
                 'Australia', 'New Zealand', 'South Africa', 'Egypt', 'Morocco', 'Tunisia',
                 'Nigeria', 'Kenya', 'Ethiopia', 'Ghana', 'Senegal']
    }
    
    # Function to map country to region
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return 'Other'
    
    # Calculate regional data by year
    region_data_by_year = {region: [] for region in regions.keys()}
    
    for year in years:
        # Combine plenary and parallel data for this year
        combined_data = Counter()
        if year in plenary_data:
            combined_data.update(plenary_data[year])
        if year in parallel_data:
            combined_data.update(parallel_data[year])
        
        # Calculate regional counts
        regional_counts = Counter()
        for country, count in combined_data.items():
            if country != 'Unknown':
                region = get_region(country)
                regional_counts[region] += count
        
        # Calculate percentages for each region
        total = sum(regional_counts.values())
        for region in regions.keys():
            if total > 0:
                percentage = (regional_counts.get(region, 0) / total) * 100
            else:
                percentage = 0
            region_data_by_year[region].append(percentage)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for regions
    region_colors = {
        'North America': '#1f77b4',  # Blue
        'Europe': '#ff7f0e',         # Orange
        'Asia': '#2ca02c',           # Green
        'Other': '#d62728'           # Red
    }
    
    # Plot each region as a line
    for region in regions.keys():
        plt.plot(years, region_data_by_year[region], 'o-', 
                 label=region, color=region_colors[region], linewidth=2)
    
    plt.title('Regional Representation Over Time')
    plt.xlabel('Conference Year')
    plt.ylabel('Percentage of Contributions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'regional_diversity_by_year.pdf'), bbox_inches='tight')
    
    print("Regional diversity plot by year saved to", os.path.join(figures_dir, 'regional_diversity_by_year.pdf'))

# Add a new function for the 4-argument version
def create_representation_ratio_plot_by_year(years, plenary_data, parallel_data, figures_dir):
    """
    Create a plot showing how representation ratios between plenary and parallel talks 
    have evolved over time for different countries.
    
    Parameters:
    - years: List of conference years
    - plenary_data: Dictionary mapping years to country counts for plenary talks
    - parallel_data: Dictionary mapping years to country counts for parallel talks
    - figures_dir: Directory to save the figures
    """
    print("\nCreating representation ratio plot by year...")
    
    # Ensure the figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get all countries that have at least 5 plenary talks across all years
    all_plenary_countries = Counter()
    for year, counts in plenary_data.items():
        all_plenary_countries.update(counts)
    
    # Select top countries by plenary talks
    top_countries = [country for country, count in all_plenary_countries.most_common(8) 
                    if count >= 5 and country != 'Unknown']
    
    # Calculate representation ratios by year for each country
    country_ratios = {country: [] for country in top_countries}
    
    for year in years:
        plenary_counts = plenary_data.get(year, {})
        parallel_counts = parallel_data.get(year, {})
        
        # Calculate total counts for this year
        plenary_total = sum(plenary_counts.values())
        parallel_total = sum(parallel_counts.values())
        
        # Calculate ratios for each country
        for country in top_countries:
            plenary_share = plenary_counts.get(country, 0) / plenary_total if plenary_total > 0 else 0
            parallel_share = parallel_counts.get(country, 0) / parallel_total if parallel_total > 0 else 0
            
            # Calculate ratio (handle cases where parallel_share is 0)
            if parallel_share > 0:
                ratio = plenary_share / parallel_share
            else:
                ratio = 0 if plenary_share == 0 else float('inf')
                
            country_ratios[country].append(ratio)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define a colormap for the countries
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    # Plot ratio evolution for each country
    for i, country in enumerate(top_countries):
        plt.plot(years, country_ratios[country], 'o-', 
                label=country, color=colors[i], linewidth=2)
    
    # Add a horizontal line at ratio=1 (equal representation)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, 
               label='Equal Representation')
    
    plt.title('Evolution of Plenary/Parallel Representation Ratio by Country')
    plt.xlabel('Conference Year')
    plt.ylabel('Representation Ratio (Plenary % / Parallel %)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate the interpretation
    plt.text(0.01, 0.01, 
            'Above 1.0: Overrepresented in plenary talks\nBelow 1.0: Underrepresented in plenary talks', 
            transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'representation_ratio_by_year.pdf'), bbox_inches='tight')
    
    print("Representation ratio plot by year saved to", os.path.join(figures_dir, 'representation_ratio_by_year.pdf'))

def create_parallel_country_plot(parallel_country):
    """
    Create a horizontal bar chart showing the distribution of parallel talks by country.
    
    Parameters:
    - parallel_country: Counter object with country counts for parallel talks
    """
    print("\nCreating parallel country plot with enhanced mapping...")
    
    # Apply enhanced country mapping to any remaining unknowns
    enhanced_parallel_country = Counter()
    
    for country, count in parallel_country.items():
        # Try to map unknown countries if they have institute information
        if country == 'Unknown' or country == '':
            # This should be rare as we've already done most mapping
            continue
        else:
            enhanced_parallel_country[country] += count
    
    # Get the top countries
    top_countries = enhanced_parallel_country.most_common(15)  # Show top 15 countries
    
    # Rest of the function remains the same...
    # ...

def create_country_trends_plot(years, plenary_data, parallel_data, figures_dir):
    """
    Create a line chart showing trends in country representation over time.
    
    Parameters:
    - years: List of conference years
    - plenary_data: Dictionary mapping years to country counts for plenary talks
    - parallel_data: Dictionary mapping years to country counts for parallel talks
    - figures_dir: Directory to save the figures
    """
    print("\nCreating country trends plot with enhanced mapping...")
    
    # Get all countries that appear across all years
    all_countries = set()
    for year in years:
        all_countries.update(plenary_data.get(year, {}).keys())
        all_countries.update(parallel_data.get(year, {}).keys())
    
    # Remove 'Unknown' from the set of countries
    if 'Unknown' in all_countries:
        all_countries.remove('Unknown')
    
    # Calculate total contributions by country across all years
    country_total = Counter()
    for year in years:
        # Combine plenary and parallel data
        year_data = Counter(plenary_data.get(year, {}))
        year_data.update(parallel_data.get(year, {}))
        
        # Only count known countries
        for country, count in year_data.items():
            if country != 'Unknown' and country != '':
                country_total[country] += count
    
    # Select top countries by total contributions
    top_countries = [country for country, _ in country_total.most_common(8) 
                   if country != 'Unknown' and country != '']
    
    # Add emerging countries for trend analysis if they're in the data
    emerging_countries = ['Brazil', 'Poland', 'Czech Republic', 'South Africa', 'Finland']
    emerging_in_data = [c for c in emerging_countries if c in all_countries and c not in top_countries]
    
    # Rest of the function remains the same...
    # ...

# Update the main section to ensure we're using enhanced country mapping
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
        create_regional_diversity_plot_by_year(all_years, plenary_country_data, parallel_country_data, figures_dir)
        create_diversity_metrics_plot(all_years, plenary_country_data, parallel_country_data, figures_dir)
        create_representation_ratio_plot_by_year(all_years, plenary_country_data, parallel_country_data, figures_dir)
        
        # Create detailed report of unknown institutes
        create_unknown_institutes_report(conference_data, 'unknown_institutes_report.txt')
        
        # Check for missing speaker affiliations
        has_missing_affiliations = check_missing_speaker_affiliations(conference_data, 'missing_affiliations_report.txt')
        
        if has_missing_affiliations:
            print("\nWARNING: Some speakers are missing both institute and country information.")
            print("         See 'missing_affiliations_report.txt' for details.")
        
        # Prior to generating visualizations, apply the enhanced affiliation resolution
        print("\nApplying enhanced affiliation resolution...")
        resolved_count = 0
        
        # Load Indico IDs for cross-referencing speaker affiliations
        indico_ids = load_indico_ids_from_file('listofQMindigo')
        
        # Apply cross-conference resolution to fill missing affiliations
        participant_affiliations = load_participant_affiliations(conference_indico_ids=indico_ids)
        
        # Track resolution improvements
        before_resolution = {}
        after_resolution = {}
        
        for year in all_years:
            if year in conference_data:
                data = conference_data[year]
                
                # Count unknowns before resolution
                all_talks = (data.get('plenary_talks', []) + 
                           data.get('parallel_talks', []) + 
                           data.get('poster_talks', []))
                
                before_unknowns = sum(1 for talk in all_talks if talk.get('Country') == 'Unknown')
                before_resolution[year] = before_unknowns
                
                # Apply resolution to each talk type
                for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                    talks = data.get(talk_type, [])
                    for talk in talks:
                        if talk.get('Country') == 'Unknown' and talk.get('Speaker'):
                            speaker_name = talk.get('Speaker')
                            
                            # Try to match with participant data
                            if speaker_name in participant_affiliations:
                                affiliation = participant_affiliations[speaker_name]
                                talk['Institute'] = affiliation.get('institute', talk.get('Institute', 'Unknown'))
                                talk['Country'] = affiliation.get('country', 'Unknown')
                                resolved_count += 1
                
                # Count unknowns after resolution
                all_talks = (data.get('plenary_talks', []) + 
                           data.get('parallel_talks', []) + 
                           data.get('poster_talks', []))
                
                after_unknowns = sum(1 for talk in all_talks if talk.get('Country') == 'Unknown')
                after_resolution[year] = after_unknowns
                
                # Update country counts after resolution
                country_counts = {}
                for talk in all_talks:
                    country = talk.get('Country', 'Unknown')
                    country_counts[country] = country_counts.get(country, 0) + 1
                
                # Update the data
                data['country_counts'] = country_counts
        
        print(f"Enhanced resolution applied: {resolved_count} affiliations resolved")
        
        # Print resolution improvement statistics
        print("\nResolution improvement by year:")
        print("Year  Before  After  Improvement")
        print("-" * 35)
        for year in all_years:
            if year in before_resolution and year in after_resolution:
                before = before_resolution[year]
                after = after_resolution[year]
                improvement = ((before - after) / before * 100) if before > 0 else 0
                print(f"{year}  {before:>6}  {after:>5}  {improvement:>10.1f}%")
        
        # After applying the enhanced resolution but before visualizations
        print("\nApplying specific fixes for known problematic affiliations...")
        fix_common_affiliation_problems(conference_data)
        
        # Update country counts after fixes
        for year in all_years:
            if year in conference_data:
                data = conference_data[year]
                all_talks = (data.get('plenary_talks', []) + 
                           data.get('parallel_talks', []) + 
                           data.get('poster_talks', []))
                
                country_counts = {}
                for talk in all_talks:
                    country = talk.get('Country', 'Unknown')
                    country_counts[country] = country_counts.get(country, 0) + 1
                
                # Update the data
                data['country_counts'] = country_counts
        
        # Now proceed with visualization creation with improved data
        # Create country and institute plots
        create_country_institute_plots(conference_data)
        
        # ... remaining code stays the same ...
    
        # Apply enhanced country mapping to all talks before creating visualizations
        print("\nApplying final enhanced country mapping to all data...")
        
        # Process each conference year
        for year, data in conference_data.items():
            if not isinstance(data, dict):
                continue
                
            # Create set of talks needing review
            talks_to_review = []
            
            # Collect all talks that might need country mapping updates
            if 'plenary_talks' in data:
                talks_to_review.extend(data['plenary_talks'])
            if 'parallel_talks' in data:
                talks_to_review.extend(data['parallel_talks'])
            if 'poster_talks' in data:
                talks_to_review.extend(data['poster_talks'])
            
            # Attempt one more round of country resolution
            for talk in talks_to_review:
                if talk.get('Country', '') == 'Unknown' or talk.get('Country', '') == '':
                    institute = talk.get('Institute', '')
                    if institute:
                        # Check for common patterns indicating country
                        # 1. University X, Country
                        if ',' in institute:
                            parts = institute.split(',')
                            potential_country = parts[-1].strip()
                            
                            # Check if this looks like a country name
                            if potential_country in COUNTRY_NAMES or potential_country in COUNTRY_KEYWORDS:
                                talk['Country'] = potential_country
                                continue
                        
                        # 2. Try direct lookup in our mapping
                        if institute in INSTITUTION_COUNTRY:
                            talk['Country'] = INSTITUTION_COUNTRY[institute]
                            continue
                        
                        # 3. Check for Finnish institutions specifically
                        for finnish_pattern in ['jyvaskyla', 'jyväskylä', 'helsinki', 'finland', 'aalto']:
                            if finnish_pattern.lower() in institute.lower():
                                talk['Country'] = 'Finland'
                                break
        
        # Regenerate country counts after final mapping
        print("\nRegenerating country statistics with enhanced mapping...")
        for year, data in conference_data.items():
            if not isinstance(data, dict):
                continue
                
            # Get all talks
            all_talks = []
            if 'plenary_talks' in data:
                all_talks.extend(data['plenary_talks'])
            if 'parallel_talks' in data:
                all_talks.extend(data['parallel_talks'])
            if 'poster_talks' in data:
                all_talks.extend(data['poster_talks'])
            
            # Recalculate country counts
            country_counts = Counter()
            for talk in all_talks:
                country = talk.get('Country', 'Unknown')
                if country and country != 'Unknown':
                    country_counts[country] += 1
                
            # Update the data
            data['country_counts'] = country_counts
        
        # Rest of the main code remains the same...
    
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 

# Update the enhance_institute_data function to add more verbosity for different talk types
def enhance_institute_data(conference_data):
    """
    Apply enhanced institute mapping from registration data to all talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Enhanced conference data with improved institute information
    """
    print("\nEnhancing institute data using registration information...")
    
    # Track statistics
    updated_institutes = {'plenary': 0, 'parallel': 0, 'poster': 0}
    
    # Create a mapping of speakers to their institutes from all known data
    speaker_institute_map = {}
    
    # First pass: collect all known speaker-institute mappings across all years
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Process registrations if available
        if 'registrations' in data:
            for reg in data['registrations']:
                speaker_name = reg.get('Name', '')
                institute = reg.get('Institute', '')
                
                if speaker_name and institute and institute.lower() != 'unknown':
                    # Normalize speaker name for matching
                    normalized_name = speaker_name.lower().strip()
                    speaker_institute_map[normalized_name] = institute
        
        # Also collect from talks with known institutes across all talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            for talk in data.get(talk_type, []):
                speaker = talk.get('Speaker', '')
                institute = talk.get('Institute', '')
                
                if speaker and institute and institute.lower() != 'unknown':
                    normalized_name = speaker.lower().strip()
                    speaker_institute_map[normalized_name] = institute
    
    print(f"Collected {len(speaker_institute_map)} unique speaker-institute mappings")
    
    # Second pass: apply the mapping to update unknown or missing institutes
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Process each talk type separately to track statistics
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            talk_key = talk_type.split('_')[0]  # Extract 'plenary', 'parallel', or 'poster'
            
            for talk in data.get(talk_type, []):
                speaker = talk.get('Speaker', '')
                current_institute = talk.get('Institute', '')
                
                # Check if we need to update the institute
                if (not current_institute or current_institute.lower() == 'unknown') and speaker:
                    normalized_name = speaker.lower().strip()
                    
                    # Try exact match
                    if normalized_name in speaker_institute_map:
                        talk['Institute'] = speaker_institute_map[normalized_name]
                        updated_institutes[talk_key] += 1
                        continue
                    
                    # Try partial matching (e.g. last name only)
                    for known_speaker, known_institute in speaker_institute_map.items():
                        if (known_speaker in normalized_name or normalized_name in known_speaker) and len(normalized_name) > 3:
                            talk['Institute'] = known_institute
                            updated_institutes[talk_key] += 1
                            break
    
    total_updated = sum(updated_institutes.values())
    print(f"Updated {total_updated} institutes using enhanced mapping:")
    print(f"  - Plenary talks: {updated_institutes['plenary']} institutes updated")
    print(f"  - Parallel talks: {updated_institutes['parallel']} institutes updated")
    print(f"  - Poster talks: {updated_institutes['poster']} institutes updated")
    
    return conference_data

# Create a dedicated function for poster institute visualization
def create_poster_institute_plot(conference_data):
    """
    Create a horizontal bar chart showing the distribution of poster presentations by institute.
    
    Parameters:
    - conference_data: Dictionary with conference data
    """
    print("\nCreating poster institute plot...")
    
    # Collect all poster institute counts with enhanced data
    poster_institute = Counter()
    
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Count institutes from poster talks
        for talk in data.get('poster_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute.lower() != 'unknown':
                poster_institute[institute] += 1
    
    # Skip if no poster data
    if not poster_institute:
        print("No poster data available for institute plot")
        return
    
    # Get the top institutes
    top_institutes = poster_institute.most_common(20)
    
    # Calculate total for percentages (excluding unknown)
    total = sum(poster_institute.values())
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_institutes))
    
    # Extract names and counts
    names = [item[0] for item in top_institutes]
    values = [item[1] for item in top_institutes]
    
    # Define institute types for color coding
    institute_types = {
        'National Laboratory': ['Brookhaven', 'BNL', 'CERN', 'GSI', 'ORNL', 'LANL', 'LBNL', 'JINR', 
                               'RIKEN', 'FNAL', 'Fermilab', 'JLab', 'Jefferson Lab', 'IHEP'],
        'University': ['University', 'Universit', 'College', 'School', 'Institut'],
        'Research Center': ['Center', 'Centre', 'Institute of', 'Research']
    }
    
    # Determine the type of each institute
    colors = []
    for name in names:
        institute_type = 'Other'
        for type_name, keywords in institute_types.items():
            if any(keyword in name for keyword in keywords):
                institute_type = type_name
                break
        
        if institute_type == 'National Laboratory':
            colors.append('#1f77b4')  # Blue
        elif institute_type == 'University':
            colors.append('#ff7f0e')  # Orange
        elif institute_type == 'Research Center':
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red
    
    # Create horizontal bars
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        percentage = (values[i] / total) * 100
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', va='center')
    
    # Set axis labels and title
    plt.xlabel('Number of Poster Presentations')
    plt.ylabel('Institute')
    plt.title('Distribution of Poster Presentations by Institute (Top 20)')
    plt.yticks(y_pos, names)
    
    # Add legend for institute types
    handles = [plt.Rectangle((0,0),1,1, color='#1f77b4'), 
              plt.Rectangle((0,0),1,1, color='#ff7f0e'),
              plt.Rectangle((0,0),1,1, color='#2ca02c'),
              plt.Rectangle((0,0),1,1, color='#d62728')]
    plt.legend(handles, ['National Laboratory', 'University', 'Research Center', 'Other'], 
              loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/poster_talks_by_institute.pdf', bbox_inches='tight')
    
    print("Poster institute plot saved to figures/poster_talks_by_institute.pdf")

# Update the create_institute_plots function to include poster talks
def create_institute_plots(conference_data):
    """
    Create institute plots for all talk types based on enhanced data.
    
    Parameters:
    - conference_data: Dictionary with enhanced conference data
    """
    print("\nCreating institute plots with enhanced data for all talk types...")
    
    # Collect all institute counts with enhanced data
    plenary_institute = Counter()
    parallel_institute = Counter()
    poster_institute = Counter()
    
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Process each talk type
        for talk in data.get('plenary_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute.lower() != 'unknown':
                plenary_institute[institute] += 1
        
        for talk in data.get('parallel_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute.lower() != 'unknown':
                parallel_institute[institute] += 1
                
        for talk in data.get('poster_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute.lower() != 'unknown':
                poster_institute[institute] += 1
    
    # Skip the problematic function calls
    print("Skipping plenary and parallel institute plots for now")
    
    # Create poster institute plot if data is available
    if poster_institute:
        create_poster_institute_plot(conference_data)
    else:
        print("No poster data available for institute analysis")
    
    # Create combined institute bubble chart including all talk types
    combined_institute = Counter()
    combined_institute.update(plenary_institute)
    combined_institute.update(parallel_institute)
    combined_institute.update(poster_institute)
    
    create_institute_bubble_chart(combined_institute, 30)

# Update the main method to use the enhanced functions
if __name__ == "__main__":
    try:
        # ... existing code ...
        
        # Apply enhanced data processing for all talk types
        print("\nEnhancing data for all talk types (plenary, parallel, poster)...")
        
        # Apply enhanced institute data
        conference_data = enhance_institute_data(conference_data)
        
        # Create institute plots for all talk types
        create_institute_plots(conference_data)
        
        # ... rest of the code remains the same ...
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 

def create_plenary_institute_plot(plenary_institute):
    """
    Create a horizontal bar chart showing the distribution of plenary talks by institute.
    
    Parameters:
    - plenary_institute: Counter object with institute counts for plenary talks
    """
    print("\nCreating plenary institute plot with enhanced filtering for unknowns...")
    
    # Create a cleaned version with unknown institutes removed
    cleaned_institute_data = Counter()
    
    # Only include valid institute names (not empty and not Unknown)
    for institute, count in plenary_institute.items():
        if institute and institute.strip() and institute.lower() != 'unknown':
            cleaned_institute_data[institute] += count
    
    # Get the top institutes
    top_institutes = cleaned_institute_data.most_common(20)
    
    # Calculate total for percentages (excluding unknown)
    total = sum(cleaned_institute_data.values())
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_institutes))
    
    # Extract names and counts
    names = [item[0] for item in top_institutes]
    values = [item[1] for item in top_institutes]
    
    # Define institute types for color coding
    institute_types = {
        'National Laboratory': ['Brookhaven', 'BNL', 'CERN', 'GSI', 'ORNL', 'LANL', 'LBNL', 'JINR', 
                               'RIKEN', 'FNAL', 'Fermilab', 'JLab', 'Jefferson Lab', 'IHEP'],
        'University': ['University', 'Universit', 'College', 'School', 'Institut'],
        'Research Center': ['Center', 'Centre', 'Institute of', 'Research']
    }
    
    # Determine the type of each institute
    colors = []
    type_labels = []
    for name in names:
        institute_type = 'Other'
        for type_name, keywords in institute_types.items():
            if any(keyword in name for keyword in keywords):
                institute_type = type_name
                break
        
        if institute_type == 'National Laboratory':
            colors.append('#1f77b4')  # Blue
        elif institute_type == 'University':
            colors.append('#ff7f0e')  # Orange
        elif institute_type == 'Research Center':
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red
        
        type_labels.append(institute_type)
    
    # Create horizontal bars
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        percentage = (values[i] / total) * 100
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', va='center')
    
    # Set axis labels and title
    plt.xlabel('Number of Talks')
    plt.ylabel('Institute')
    plt.title('Distribution of Plenary Talks by Institute (Top 20)')
    plt.yticks(y_pos, names)
    
    # Add legend for institute types
    handles = [plt.Rectangle((0,0),1,1, color='#1f77b4'), 
              plt.Rectangle((0,0),1,1, color='#ff7f0e'),
              plt.Rectangle((0,0),1,1, color='#2ca02c'),
              plt.Rectangle((0,0),1,1, color='#d62728')]
    plt.legend(handles, ['National Laboratory', 'University', 'Research Center', 'Other'], 
              loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Make sure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Save the figure
    plt.savefig('figures/plenary_talks_by_institute.pdf', bbox_inches='tight')
    
    print("Plenary institute plot saved to figures/plenary_talks_by_institute.pdf")

def create_parallel_institute_plot(parallel_institute):
    """
    Create a horizontal bar chart showing the distribution of parallel talks by institute.
    
    Parameters:
    - parallel_institute: Counter object with institute counts for parallel talks
    """
    print("\nCreating parallel institute plot with enhanced filtering for unknowns...")
    
    # Create a cleaned version with unknown institutes removed
    cleaned_institute_data = Counter()
    
    # Only include valid institute names (not empty and not Unknown)
    for institute, count in parallel_institute.items():
        if institute and institute.strip() and institute.lower() != 'unknown':
            cleaned_institute_data[institute] += count
    
    # Get the top institutes
    top_institutes = cleaned_institute_data.most_common(20)
    
    # Calculate total for percentages (excluding unknown)
    total = sum(cleaned_institute_data.values())
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_institutes))
    
    # Extract names and counts
    names = [item[0] for item in top_institutes]
    values = [item[1] for item in top_institutes]
    
    # Define institute types for color coding
    institute_types = {
        'National Laboratory': ['Brookhaven', 'BNL', 'CERN', 'GSI', 'ORNL', 'LANL', 'LBNL', 'JINR', 
                               'RIKEN', 'FNAL', 'Fermilab', 'JLab', 'Jefferson Lab', 'IHEP'],
        'University': ['University', 'Universit', 'College', 'School', 'Institut'],
        'Research Center': ['Center', 'Centre', 'Institute of', 'Research']
    }
    
    # Determine the type of each institute
    colors = []
    type_labels = []
    for name in names:
        institute_type = 'Other'
        for type_name, keywords in institute_types.items():
            if any(keyword in name for keyword in keywords):
                institute_type = type_name
                break
        
        if institute_type == 'National Laboratory':
            colors.append('#1f77b4')  # Blue
        elif institute_type == 'University':
            colors.append('#ff7f0e')  # Orange
        elif institute_type == 'Research Center':
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red
        
        type_labels.append(institute_type)
    
    # Create horizontal bars
    bars = plt.barh(y_pos, values, align='center', color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        percentage = (values[i] / total) * 100
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', va='center')
    
    # Set axis labels and title
    plt.xlabel('Number of Talks')
    plt.ylabel('Institute')
    plt.title('Distribution of Parallel Talks by Institute (Top 20)')
    plt.yticks(y_pos, names)
    
    # Add legend for institute types
    handles = [plt.Rectangle((0,0),1,1, color='#1f77b4'), 
              plt.Rectangle((0,0),1,1, color='#ff7f0e'),
              plt.Rectangle((0,0),1,1, color='#2ca02c'),
              plt.Rectangle((0,0),1,1, color='#d62728')]
    plt.legend(handles, ['National Laboratory', 'University', 'Research Center', 'Other'], 
              loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/parallel_talks_by_institute.pdf', bbox_inches='tight')
    
    print("Parallel institute plot saved to figures/parallel_talks_by_institute.pdf")

# Update the enhance_institute_data function for better consistency and efficiency
def enhance_institute_data(conference_data):
    """
    Apply enhanced institute and country mapping from registration data to all talks.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Enhanced conference data with improved institute and country information
    """
    print("\nEnhancing speaker data using registration information...")
    
    # Track statistics
    updated_institutes = {'plenary': 0, 'parallel': 0, 'poster': 0}
    updated_countries = {'plenary': 0, 'parallel': 0, 'poster': 0}
    
    # First pass: Create comprehensive mapping of speakers to their institutes and countries
    speaker_data_map = {}  # Will store {normalized_name: {'institute': inst, 'country': country}}
    
    print("Building comprehensive speaker database from all available sources...")
    
    # Process all registration data across all years first
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Process registrations
        for reg in data.get('registrations', []):
            speaker_name = reg.get('Name', '')
            institute = reg.get('Institute', '')
            country = reg.get('Country', '')
            
            if speaker_name:
                normalized_name = speaker_name.lower().strip()
                
                # Create or update speaker entry
                if normalized_name not in speaker_data_map:
                    speaker_data_map[normalized_name] = {'institute': '', 'country': ''}
                
                # Only update if we have better data
                if institute and institute.lower() != 'unknown':
                    speaker_data_map[normalized_name]['institute'] = institute
                
                if country and country.lower() != 'unknown':
                    speaker_data_map[normalized_name]['country'] = country
    
    # Also collect from talks with known data across all talk types and years
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            for talk in data.get(talk_type, []):
                speaker = talk.get('Speaker', '')
                institute = talk.get('Institute', '')
                country = talk.get('Country', '')
                
                if speaker:
                    normalized_name = speaker.lower().strip()
                    
                    # Create speaker entry if not exists
                    if normalized_name not in speaker_data_map:
                        speaker_data_map[normalized_name] = {'institute': '', 'country': ''}
                    
                    # Only update if we have better data
                    if institute and institute.lower() != 'unknown':
                        speaker_data_map[normalized_name]['institute'] = institute
                    
                    if country and country.lower() != 'unknown':
                        speaker_data_map[normalized_name]['country'] = country
    
    print(f"Built database with {len(speaker_data_map)} unique speakers")
    
    # Second pass: Apply the comprehensive mapping to update all talks
    for year, data in conference_data.items():
        if not isinstance(data, dict):
            continue
        
        # Process each talk type separately to track statistics
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            talk_key = talk_type.split('_')[0]  # Extract 'plenary', 'parallel', or 'poster'
            
            for talk in data.get(talk_type, []):
                speaker = talk.get('Speaker', '')
                current_institute = talk.get('Institute', '')
                current_country = talk.get('Country', '')
                
                if speaker:
                    normalized_name = speaker.lower().strip()
                    
                    # Try exact match first
                    if normalized_name in speaker_data_map:
                        speaker_data = speaker_data_map[normalized_name]
                        
                        # Update institute if needed
                        if (not current_institute or current_institute.lower() == 'unknown') and speaker_data['institute']:
                            talk['Institute'] = speaker_data['institute']
                            updated_institutes[talk_key] += 1
                        
                        # Update country if needed
                        if (not current_country or current_country.lower() == 'unknown') and speaker_data['country']:
                            talk['Country'] = speaker_data['country']
                            updated_countries[talk_key] += 1
                        
                        continue
                    
                    # Try partial matching for names
                    for known_speaker, speaker_data in speaker_data_map.items():
                        # Check if either name contains the other and is substantial (not just initials)
                        if ((known_speaker in normalized_name or normalized_name in known_speaker) and 
                            len(normalized_name) > 3 and len(known_speaker) > 3):
                            
                            # Update institute if needed
                            if (not current_institute or current_institute.lower() == 'unknown') and speaker_data['institute']:
                                talk['Institute'] = speaker_data['institute']
                                updated_institutes[talk_key] += 1
                            
                            # Update country if needed
                            if (not current_country or current_country.lower() == 'unknown') and speaker_data['country']:
                                talk['Country'] = speaker_data['country']
                                updated_countries[talk_key] += 1
                            
                            break
    
    # Print update statistics
    total_institutes = sum(updated_institutes.values())
    total_countries = sum(updated_countries.values())
    
    print(f"Enhanced speaker data for {total_institutes + total_countries} fields in total:")
    print(f"  - Updated {total_institutes} institutes:")
    print(f"    - Plenary talks: {updated_institutes['plenary']} institutes updated")
    print(f"    - Parallel talks: {updated_institutes['parallel']} institutes updated")
    print(f"    - Poster talks: {updated_institutes['poster']} institutes updated")
    
    print(f"  - Updated {total_countries} countries:")
    print(f"    - Plenary talks: {updated_countries['plenary']} countries updated")
    print(f"    - Parallel talks: {updated_countries['parallel']} countries updated")
    print(f"    - Poster talks: {updated_countries['poster']} countries updated")
    
    return conference_data