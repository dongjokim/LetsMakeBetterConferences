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


def extract_affiliation_info(talk_data):
    """
    Extract institute and country information more effectively from talk data.
    Handles various formatting patterns and edge cases.
    Uses global PARTICIPANT_AFFILIATIONS and INSTITUTE_COUNTRY_MAPPINGS if available.
    
    Parameters:
    - talk_data: Dictionary containing talk information
    
    Returns:
    - Tuple of (institute, country)
    """
    global PARTICIPANT_AFFILIATIONS, INSTITUTE_COUNTRY_MAPPINGS
    
    institute = talk_data.get('Institute', '').strip()
    country = talk_data.get('Country', '').strip()
    speaker = talk_data.get('Speaker', '').strip()
    
    # If both fields are filled, no extraction needed
    if institute and institute.lower() != 'unknown' and country and country.lower() != 'unknown':
        return institute, country
    
    # First try to match with participant data if available
    if 'PARTICIPANT_AFFILIATIONS' in globals() and PARTICIPANT_AFFILIATIONS and speaker:
        # Try exact match first
        if speaker in PARTICIPANT_AFFILIATIONS:
            affiliation = PARTICIPANT_AFFILIATIONS[speaker]
            
            # Update institute if needed
            if (not institute or institute.lower() == 'unknown') and affiliation['Institute'] != 'Unknown':
                institute = affiliation['Institute']
            
            # Update country if needed
            if (not country or country.lower() == 'unknown') and affiliation['Country'] != 'Unknown':
                country = affiliation['Country']
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
                    
                    # Update country if needed
                    if (not country or country.lower() == 'unknown') and affiliation['Country'] != 'Unknown':
                        country = affiliation['Country']
                    
                    break
    
    # Check if we have a country mapping for this institute
    if institute and (not country or country.lower() == 'unknown') and INSTITUTE_COUNTRY_MAPPINGS:
        # Try exact match
        if institute in INSTITUTE_COUNTRY_MAPPINGS:
            country = INSTITUTE_COUNTRY_MAPPINGS[institute]
        else:
            # Try case-insensitive match
            institute_lower = institute.lower()
            for known_inst, known_country in INSTITUTE_COUNTRY_MAPPINGS.items():
                if known_inst.lower() == institute_lower:
                    country = known_country
                    break
    
    # Rest of the function remains the same...



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
    """
    Load participant data from CSV files
    """
    participant_data = {}
    data_dir = "data"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} not found. No participant data will be loaded.")
        return participant_data
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('participants_') and f.endswith('.csv')]
    
    if not csv_files:
        print("No participant CSV files found in data directory.")
        return participant_data
    
    print(f"Found {len(csv_files)} CSV participant data files")
    
    # Process each CSV file
    import csv
    for csv_file in csv_files:
        # Extract year from filename (participants_YYYY.csv)
        match = re.search(r'participants_(\d{4})\.csv', csv_file)
        if match:
            year = match.group(1)
            file_path = os.path.join(data_dir, csv_file)
            
            try:
                # Read CSV file
                participants = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if len(row) >= 3:  # Name, Affiliation, Country, Year
                            participant = {
                                'name': row[0].strip('"'),
                                'affiliation': row[1].strip('"'),
                                'country': row[2].strip('"') if row[2].strip() else "Unknown"
                            }
                            participants.append(participant)
                
                participant_data[year] = participants
                print(f"  Loaded {len(participants)} participants for QM{year} from CSV")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    return participant_data

def fix_unknown_institutes_from_participants(conference_data, participant_data):
    """
    Fix unknown institutes and countries for speakers using participant data.
    First map institutes, then handle countries separately.
    
    Parameters:
    - conference_data: Dictionary with conference data
    - participant_data: Dictionary mapping years to lists of participants
    
    Returns:
    - Tuple of (institute_fixes, country_fixes)
    """
    print("\nFixing unknown institutes and countries using participant data...")
    
    institute_fixes = 0
    country_fixes = 0
    
    for year, data in conference_data.items():
        if year not in participant_data:
            continue
            
        # Create a lookup dictionary for faster matching
        participant_lookup = {}
        for participant in participant_data[year]:
            name = participant.get('name', '').lower()
            if name:
                participant_lookup[name] = participant
        
        # Process all talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type not in data:
                continue
                
            for talk in data[talk_type]:
                speaker = talk.get('Speaker', '').lower()
                if not speaker:
                    continue
                
                # First pass: Fix institutes
                institute_unknown = not talk.get('Institute') or talk.get('Institute').lower() == 'unknown'
                if institute_unknown:
                    # Try exact match first
                    if speaker in participant_lookup:
                        participant = participant_lookup[speaker]
                        if participant.get('affiliation'):
                            talk['Institute'] = participant['affiliation']
                            institute_fixes += 1
                    else:
                        # Try fuzzy matching for institute
                        for p_name, participant in participant_lookup.items():
                            if (speaker in p_name or p_name in speaker) and participant.get('affiliation'):
                                talk['Institute'] = participant['affiliation']
                                institute_fixes += 1
                                break
                
                # Second pass: Fix countries
                country_unknown = not talk.get('Country') or talk.get('Country').lower() == 'unknown'
                if country_unknown:
                    # Try exact match first
                    if speaker in participant_lookup:
                        participant = participant_lookup[speaker]
                        if participant.get('country'):
                            talk['Country'] = participant['country']
                            country_fixes += 1
                    else:
                        # Try fuzzy matching for country
                        for p_name, participant in participant_lookup.items():
                            if (speaker in p_name or p_name in speaker) and participant.get('country'):
                                talk['Country'] = participant['country']
                                country_fixes += 1
                                break
    
    print(f"Applied {institute_fixes} institute fixes and {country_fixes} country fixes using participant data")
    return (institute_fixes, country_fixes)

def fetch_and_analyze_conferences():
    """
    Main function to fetch and analyze conference data.
    """
    try:
        # Load Indico IDs from file
        indico_ids = load_indico_ids_from_file('listofQMindigo')
        
        # Process each conference
        conferences = []
        for year, indico_id in indico_ids.items():
            conferences.append((year, indico_id))
        
        # Sort conferences by year
        conferences.sort(key=lambda x: x[0])
        
        # Process each conference
        conference_data = {}  # Initialize as an empty dictionary
        
        for year, indico_id in conferences:
            print(f"\nProcessing QM{year} (Indico ID: {indico_id})...")
            
            # Check if we already have processed data
            processed_file = f"data/processed/qm{year}_processed.json"
            
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"  Loaded processed data from {processed_file}")
                    
                    conference_data[year] = {
                        'all_talks': data['all_talks'],
                        'plenary_talks': data['plenary_talks'],
                        'parallel_talks': data['parallel_talks'],
                        'poster_talks': data['poster_talks'],
                        'flash_talks': data.get('flash_talks', []),
                        'metadata': data.get('metadata', {})
                    }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  Could not load processed data: {e}")
                continue
        
        # ====== PRE-PROCESSING PHASE ======
        print("\n===== BEGINNING PRE-PROCESSING PHASE =====")
        
        # Load participant data first (before ANY analysis)
        print("\nLoading participant data...")
        participant_data = load_participant_data()
        
        # Store participant data as PARTICIPANT_AFFILIATIONS (global)
        global PARTICIPANT_AFFILIATIONS
        PARTICIPANT_AFFILIATIONS = {}
        
        # Flatten participant data into a usable format
        if participant_data:
            for year, participants in participant_data.items():
                for participant in participants:
                    name = participant.get('name', '')
                    if name:
                        PARTICIPANT_AFFILIATIONS[name] = {
                            'Institute': participant.get('affiliation', ''),
                            'Country': participant.get('country', '')
                        }
            print(f"Loaded {len(PARTICIPANT_AFFILIATIONS)} unique participant records")
        else:
            print("No participant data available")
            
        # Update speaker information from participant data
        if participant_data:
            conference_data = update_speaker_info_from_participant_data(conference_data, participant_data)
        else:
            print("No participant data available for updating speaker information")
        
        # Clean and standardize institute-country mappings
        print("\nCleaning institute-country mappings...")
        INSTITUTE_COUNTRY_MAPPINGS = clean_institute_country_mappings('unknown_institute_mappings.csv', 'cleaned_institute_mappings.csv')
        
        # Fix inconsistencies in institute and country data
        print("\nFixing inconsistencies in institute and country data...")
        conference_data = fix_unknown_institute_country_data(conference_data)
        
        # Fix common affiliation problems
        print("\nFixing common affiliation problems...")
        fix_common_affiliation_problems(conference_data)
        
        # Filter to only include plenary, parallel, and poster talks
        print("\nFiltering to only include plenary, parallel, and poster talks...")
        conference_data = filter_relevant_talk_types(conference_data)
        
        # ====== ANALYSIS PHASE (AFTER ALL DATA PROCESSING) ======
        print("\n===== BEGINNING ANALYSIS PHASE =====")
        print("All analyses will be performed on data with updated speaker information.")
        
        # Make sure figures directory exists
        os.makedirs('figures', exist_ok=True)
        
        # Generate country statistics
        print("\nGenerating country statistics...")
        country_counts = analyze_country_distribution(conference_data)
        
        # Generate plenary talk analysis
        print("\nAnalyzing plenary talks...")
        plenary_country, parallel_country = analyze_plenary_vs_parallel(conference_data)
        
        # Create plots
        print("\nCreating visualization plots...")
        create_plenary_country_plot(plenary_country)
        create_representation_ratio_plot(plenary_country, parallel_country)
        
        # Create regional diversity plots
        create_regional_diversity_plot(country_counts, conference_data)
        
        # Convert to dataframes for additional analysis
        print("\nConverting to dataframes for additional analysis...")
        dataframes = convert_to_dataframes(conference_data)
        
        # Create dataframe visualizations
        create_dataframe_visualizations(dataframes)
        
        # Save the updated data
        for year in conference_data:
            output_file = f"data/processed/qm{year}_processed_updated.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(conference_data[year], f, indent=2)
                print(f"Updated data saved to {output_file}")
            except Exception as e:
                print(f"Error saving updated data for {year}: {e}")
        
        print("\nProcessing complete. All data has been updated, filtered, and saved.")
        print("All statistics only include plenary, parallel, and poster talks.")
        print("All analyses were performed after updating speaker information from participant data.")
        
    except Exception as e:
        print(f"Error in fetch_and_analyze_conferences: {str(e)}")
        import traceback
        traceback.print_exc()

def create_dataframe_visualizations(dataframes):
    """
    Create additional visualizations using pandas DataFrames.
    
    Parameters:
    - dataframes: Dictionary of DataFrames for different talk types
    """
    print("\nCreating additional DataFrame-based visualizations...")
    
    # Make sure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Get the all talks DataFrame
    df_all = dataframes['all']
    
    # 1. Create a heatmap of countries by year
    try:
        print("Creating country heatmap by year...")
        plt.figure(figsize=(14, 10))
        
        # Filter out unknown countries
        df_filtered = df_all[~df_all['Country'].isin(['', 'Unknown', None])]
        
        # Get top 15 countries
        top_countries = df_filtered['Country'].value_counts().head(15).index.tolist()
        
        # Filter for just those countries
        df_top = df_filtered[df_filtered['Country'].isin(top_countries)]
        
        # Create pivot table
        pivot = pd.pivot_table(
            df_top, 
            values='Speaker', 
            index='Country',
            columns='Year',
            aggfunc='count',
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Number of Talks by Country and Year (Top 15 Countries)')
        plt.tight_layout()
        plt.savefig('figures/country_year_heatmap.pdf', bbox_inches='tight')
        print("Country heatmap saved to figures/country_year_heatmap.pdf")
    except Exception as e:
        print(f"Error creating country heatmap: {e}")
    
    # 2. Create a pie chart of institute types
    try:
        print("Creating institute type pie chart...")
        plt.figure(figsize=(10, 10))
        
        # Filter out unknown institutes
        df_filtered = df_all[~df_all['Institute'].isin(['', 'Unknown', None])]
        
        # Define institute types
        institute_types = {
            'National Laboratory': ['Brookhaven', 'BNL', 'CERN', 'GSI', 'ORNL', 'LANL', 'LBNL', 'JINR', 
                                   'RIKEN', 'FNAL', 'Fermilab', 'JLab', 'Jefferson Lab', 'IHEP'],
            'University': ['University', 'Universit', 'College', 'School', 'Institut'],
            'Research Center': ['Center', 'Centre', 'Institute of', 'Research']
        }
        
        # Function to determine institute type
        def get_institute_type(institute):
            for type_name, keywords in institute_types.items():
                if any(keyword in institute for keyword in keywords):
                    return type_name
            return 'Other'
        
        # Add institute type column
        df_filtered['InstituteType'] = df_filtered['Institute'].apply(get_institute_type)
        
        # Count by type
        type_counts = df_filtered['InstituteType'].value_counts()
        
        # Create pie chart
        plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
                startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Distribution of Talks by Institute Type')
        plt.tight_layout()
        plt.savefig('figures/institute_type_pie.pdf', bbox_inches='tight')
        print("Institute type pie chart saved to figures/institute_type_pie.pdf")
    except Exception as e:
        print(f"Error creating institute type pie chart: {e}")
    
    # 3. Create a stacked bar chart of talk types by year
    try:
        print("Creating talk type stacked bar chart by year...")
        plt.figure(figsize=(12, 8))
        
        # Create pivot table
        pivot = pd.pivot_table(
            df_all, 
            values='Speaker', 
            index='Year',
            columns='TalkType',
            aggfunc='count',
            fill_value=0
        )
        
        # Create stacked bar chart
        pivot.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Number of Talks by Type and Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Talks')
        plt.legend(title='Talk Type')
        plt.tight_layout()
        plt.savefig('figures/talk_type_by_year.pdf', bbox_inches='tight')
        print("Talk type stacked bar chart saved to figures/talk_type_by_year.pdf")
    except Exception as e:
        print(f"Error creating talk type stacked bar chart: {e}")

# Update the visualization function to include participant counts
def create_summary_statistics_plot(conference_data, filename="figs/QM_talk_statistics.pdf"):
    """
    Create a comprehensive plot of QM conference statistics with participant counts.
    """
    print("\nCreating summary statistics plot...")
    
    # Load participant data
    participant_data = load_participant_data()
    
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
            if year_str in participant_data:
                participant_data.append(len(participant_data[year_str]))
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
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_countries))
    
    # Calculate total for percentages
    total = sum(enhanced_parallel_country.values())
    
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
    plt.xlabel('Number of Parallel Talks')
    plt.title('Distribution of Parallel Talks by Country')
    plt.tight_layout()
    
    # Create legend for regions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=region)
                      for region, color in region_colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Save the figure
    plt.savefig('figures/parallel_talks_by_country.pdf', bbox_inches='tight')
    
    # Print the top countries
    print("\nTop Countries by Parallel Talks:")
    for i, (country, count) in enumerate(top_countries, 1):
        percentage = (count / total) * 100
        print(f"{i}. {country}: {count} ({percentage:.1f}%)")

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
    plt.figure(figsize=(14, 8))
    
    # Define a colormap for the countries
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries) + len(emerging_in_data)))
    
    # Plot trends for top countries
    for i, country in enumerate(top_countries):
        country_data = [plenary_data.get(year, {}).get(country, 0) + parallel_data.get(year, {}).get(country, 0)
                       for year in years]
        plt.plot(years, country_data, 'o-', label=country, color=colors[i], linewidth=2)
    
    # Plot trends for emerging countries
    for i, country in enumerate(emerging_in_data):
        country_data = [plenary_data.get(year, {}).get(country, 0) + parallel_data.get(year, {}).get(country, 0)
                       for year in years]
        plt.plot(years, country_data, 'o-', label=country, color=colors[len(top_countries) + i], linewidth=2)
    
    plt.title('Country Representation Trends Over Time')
    plt.xlabel('Conference Year')
    plt.ylabel('Number of Contributions')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'country_trends.pdf'), bbox_inches='tight')
    
    print("Country trends plot saved to", os.path.join(figures_dir, 'country_trends.pdf'))

def fetch_and_analyze_conferences():
    """
    Main function to fetch and analyze conference data.
    """
    # Load Indico IDs from file
    indico_ids = load_indico_ids_from_file('listofQMindigo')
    
    # Fetch and process contributions
    conference_data = {}
    for indico_id, year in indico_ids.items():
        contributions = fetch_and_process_contributions(indico_id, year)
        conference_data[year] = contributions
    
    # Extract country data
    plenary_country = Counter()
    parallel_country = Counter()
    poster_country = Counter()
    flash_country = Counter()
    
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    country = talk.get('Country', 'Unknown')
                    if country != 'Unknown':
                        if talk_type == 'plenary_talks':
                            plenary_country[country] += 1
                        elif talk_type == 'parallel_talks':
                            parallel_country[country] += 1
                        elif talk_type == 'poster_talks':
                            poster_country[country] += 1
                        elif talk_type == 'flash_talks':
                            flash_country[country] += 1
    
    # Create plots
    create_plenary_country_plot(plenary_country)
    create_representation_ratio_plot(plenary_country, parallel_country)
    create_regional_diversity_plot(plenary_country + parallel_country, conference_data)
    create_parallel_country_plot(parallel_country)
    
    # Create summary statistics plot
    create_summary_statistics_plot(conference_data, participant_data)
    
    # Create regional diversity plot by year
    years = sorted([year for year in conference_data.keys() if year.isdigit()])
    plenary_data = {year: {talk['Country']: 1 for talk in conference_data[year].get('plenary_talks', [])} for year in years}
    parallel_data = {year: {talk['Country']: 1 for talk in conference_data[year].get('parallel_talks', [])} for year in years}
    create_regional_diversity_plot_by_year(years, plenary_data, parallel_data, 'figures')
    
    # Create representation ratio plot by year
    create_representation_ratio_plot_by_year(years, plenary_data, parallel_data, 'figures')
    
    # Create country trends plot
    create_country_trends_plot(years, plenary_data, parallel_data, 'figures')
    
    # Create keywords plot
    create_keywords_plot(conference_data)
    
    # Create institute plots
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type in data:
                plot_talks_by_institute(data[talk_type], f"{year} {talk_type.capitalize()} Talks", f"{year}_{talk_type}_institutes.pdf")

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
        
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def clean_institute_country_mappings(filename='unknown_institute_mappings.csv', output_filename='cleaned_institute_mappings.csv'):
    """
    Clean and standardize the institute-country mappings file.
    
    Parameters:
    - filename: Path to the original CSV file with institute-country mappings
    - output_filename: Path to save the cleaned mappings
    
    Returns:
    - Dictionary mapping institute names to countries
    """
    if not os.path.exists(filename):
        print(f"Warning: Institute mappings file {filename} not found.")
        return {}
    
    # Define common country patterns
    country_patterns = {
        r'\(US\)': 'USA',
        r'\(JP\)': 'Japan',
        r'\(CN\)': 'China',
        r'\(DE\)': 'Germany',
        r'\(FR\)': 'France',
        r'\(IT\)': 'Italy',
        r'\(GB\)': 'United Kingdom',
        r'\(PL\)': 'Poland',
        r'\(RU\)': 'Russia',
        r'\(IN\)': 'India',
        r'\(NL\)': 'Netherlands',
        r'\(CH\)': 'Switzerland',
        r'\(ES\)': 'Spain',
        r'\(AT\)': 'Austria',
        r'\(DK\)': 'Denmark',
        r'\(SE\)': 'Sweden',
        r'\(NO\)': 'Norway',
        r'\(FI\)': 'Finland',
        r'\(HU\)': 'Hungary',
        r'\(CZ\)': 'Czech Republic',
        r'\(SK\)': 'Slovakia',
        r'\(RO\)': 'Romania',
        r'\(PT\)': 'Portugal',
        r'\(BR\)': 'Brazil',
        r'\(MX\)': 'Mexico',
        r'\(CL\)': 'Chile',
        r'\(KR\)': 'South Korea',
        r'\(TW\)': 'Taiwan',
        r'\(IL\)': 'Israel',
        r'\(AZ\)': 'Azerbaijan',
        r'\(RS\)': 'Serbia',
        r'USA$': 'USA',
        r'Japan$': 'Japan',
        r'China$': 'China',
        r'Germany$': 'Germany',
        r'France$': 'France',
        r'Italy$': 'Italy',
        r'United Kingdom$': 'United Kingdom',
        r'Poland$': 'Poland',
        r'Russia$': 'Russia',
        r'India$': 'India',
    }
    
    # Define institute-country mappings based on common patterns
    institute_country_map = {
        'Brookhaven': 'USA',
        'BNL': 'USA',
        'LBNL': 'USA',
        'LANL': 'USA',
        'ORNL': 'USA',
        'LLNL': 'USA',
        'CERN': 'Switzerland',
        'GSI': 'Germany',
        'DESY': 'Germany',
        'INFN': 'Italy',
        'CEA': 'France',
        'CNRS': 'France',
        'RIKEN': 'Japan',
        'KEK': 'Japan',
        'CCNU': 'China',
        'USTC': 'China',
        'SINAP': 'China',
        'VECC': 'India',
        'TIFR': 'India',
        'NIKHEF': 'Netherlands',
        'Heidelberg': 'Germany',
        'Frankfurt': 'Germany',
        'Bielefeld': 'Germany',
        'Wuppertal': 'Germany',
        'Darmstadt': 'Germany',
        'Tsukuba': 'Japan',
        'Tokyo': 'Japan',
        'Jyvaskyla': 'Finland',
        'Jyväskylä': 'Finland',
        'Warsaw': 'Poland',
        'Krakow': 'Poland',
        'Saclay': 'France',
        'Subatech': 'France',
        'Berkeley': 'USA',
        'Los Angeles': 'USA',
        'Chicago': 'USA',
        'Stony Brook': 'USA',
        'Urbana': 'USA',
        'Champaign': 'USA',
        'Davis': 'USA',
        'Houston': 'USA',
        'Michigan': 'USA',
        'Minnesota': 'USA',
        'Ohio': 'USA',
        'Texas': 'USA',
        'Yale': 'USA',
        'MIT': 'USA',
        'Columbia': 'USA',
        'Duke': 'USA',
        'Rice': 'USA',
        'Purdue': 'USA',
        'Vanderbilt': 'USA',
        'Stanford': 'USA',
        'Torino': 'Italy',
        'Catania': 'Italy',
        'Padova': 'Italy',
        'Roma': 'Italy',
        'Sapienza': 'Italy',
        'Florence': 'Italy',
        'Firenze': 'Italy',
        'Wuhan': 'China',
        'Shandong': 'China',
        'Tsinghua': 'China',
        'Fudan': 'China',
        'Peking': 'China',
        'Nanjing': 'China',
        'Shanghai': 'China',
        'Huazhong': 'China',
        'Osaka': 'Japan',
        'Kyoto': 'Japan',
        'Hiroshima': 'Japan',
        'Nagoya': 'Japan',
        'Sophia': 'Japan',
        'Nara': 'Japan',
        'Mumbai': 'India',
        'Kolkata': 'India',
        'Rajasthan': 'India',
        'Bhabha': 'India',
        'Sao Paulo': 'Brazil',
        'São Paulo': 'Brazil',
        'Santiago': 'Chile',
        'Barcelona': 'Spain',
        'Granada': 'Spain',
        'Utrecht': 'Netherlands',
        'Amsterdam': 'Netherlands',
        'Bergen': 'Norway',
        'Oslo': 'Norway',
        'Copenhagen': 'Denmark',
        'Lund': 'Sweden',
        'Bern': 'Switzerland',
        'Zurich': 'Switzerland',
        'Prague': 'Czech Republic',
        'Budapest': 'Hungary',
        'Kurchatov': 'Russia',
        'Dubna': 'Russia',
        'Belgrade': 'Serbia',
        'Bucharest': 'Romania',
        'Lisbon': 'Portugal',
        'Lisboa': 'Portugal',
        'Weizmann': 'Israel',
        'Technion': 'Israel',
        'Seoul': 'South Korea',
        'Korea': 'South Korea',
        'Pusan': 'South Korea',
        'Yonsei': 'South Korea',
        'Ewha': 'South Korea',
        'Chonnam': 'South Korea',
        'Sejong': 'South Korea',
        'Inha': 'South Korea',
    }
    
    # Load the original mappings
    import csv
    original_mappings = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            # Skip header row
            next(csv_reader, None)
            
            for row in csv_reader:
                if len(row) >= 2:
                    institute = row[0].strip()
                    country = row[1].strip()
                    original_mappings.append((institute, country))
    except Exception as e:
        print(f"Error loading institute mappings: {e}")
        return {}
    
    # Clean and standardize the mappings
    cleaned_mappings = {}
    for institute, country in original_mappings:
        if not institute or institute in ['Unknown', 'Unknown-Unknown-Unknown']:
            continue
            
        # Skip single letter entries
        if len(institute) <= 1:
            continue
            
        # If country is already provided and valid, use it
        if country and country not in ['', 'Unknown']:
            cleaned_mappings[institute] = country
            continue
            
        # Try to extract country from institute name using patterns
        country_found = False
        for pattern, country_name in country_patterns.items():
            if re.search(pattern, institute):
                cleaned_mappings[institute] = country_name
                country_found = True
                break
                
        # Try to match institute with known institutes
        if not country_found:
            for known_inst, known_country in institute_country_map.items():
                if known_inst.lower() in institute.lower():
                    cleaned_mappings[institute] = known_country
                    country_found = True
                    break
                    
        # If still no country found, mark as unknown
        if not country_found:
            cleaned_mappings[institute] = 'Unknown'
    
    # Save the cleaned mappings
    try:
        with open(output_filename, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Institute', 'Country'])
            
            for institute, country in sorted(cleaned_mappings.items()):
                csv_writer.writerow([institute, country])
                
        print(f"Cleaned institute mappings saved to {output_filename}")
    except Exception as e:
        print(f"Error saving cleaned mappings: {e}")
    
    return cleaned_mappings

# In the main function, add:
# Clean and standardize institute-country mappings
INSTITUTE_COUNTRY_MAPPINGS = clean_institute_country_mappings('unknown_institute_mappings.csv', 'cleaned_institute_mappings.csv')

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

def create_simple_data_quality_plot(year_stats, output_file):
    """
    Create a simple visualization of data quality statistics.
    """
    try:
        if not year_stats:
            print("No data available for data quality plot")
            return
            
        years = sorted(year_stats.keys())
        
        # Extract data
        total_values = [year_stats[year]['Total'] for year in years]
        unknown_institute_values = [year_stats[year]['Unknown Institute'] for year in years]
        unknown_country_values = [year_stats[year]['Unknown Country'] for year in years]
        unknown_both_values = [year_stats[year]['Unknown Both'] for year in years]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot absolute numbers
        plt.bar(years, total_values, label='Total Speakers')
        plt.bar(years, unknown_institute_values, label='Unknown Institute')
        plt.bar(years, unknown_country_values, label='Unknown Country')
        plt.bar(years, unknown_both_values, label='Unknown Both')
        
        plt.title('Data Quality by Year')
        plt.xlabel('Conference Year')
        plt.ylabel('Number of Speakers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ensure the figures directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the figure
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Data quality plot saved to {output_file}")
    except Exception as e:
        print(f"Error in create_simple_data_quality_plot: {e}")

# Add to the main function:

def update_speaker_info_from_participant_data(conference_data, participant_data):
    """
    Update speaker institute and country information using participant data loaded from CSV files.
    
    Parameters:
    - conference_data: Dictionary with conference data
    - participant_data: Dictionary mapping years to lists of participants (from load_participant_data)
    
    Returns:
    - Updated conference_data with improved speaker information
    """
    print("\nUpdating speaker information from participant data files...")
    
    # Initialize counters
    total_speakers = 0
    updated_institute = 0
    updated_country = 0
    
    # Process each conference year
    for year, data in conference_data.items():
        if year not in participant_data:
            print(f"  No participant data available for {year}")
            continue
            
        print(f"  Processing {year} with {len(participant_data[year])} participants")
        
        # Create a lookup dictionary for faster matching
        participant_lookup = {}
        for participant in participant_data[year]:
            name = participant.get('name', '').lower()
            if name:
                participant_lookup[name] = participant
        
        # Process each talk type
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'flash_talks']:
            if talk_type not in data:
                continue
                
            for talk in data[talk_type]:
                total_speakers += 1
                speaker = talk.get('Speaker', '').lower()
                if not speaker:
                    continue
                
                # Check if institute or country is unknown
                institute_unknown = not talk.get('Institute') or talk.get('Institute') == 'Unknown'
                country_unknown = not talk.get('Country') or talk.get('Country') == 'Unknown'
                
                if not institute_unknown and not country_unknown:
                    continue  # Skip if both are already known
                
                # Try exact match first
                if speaker in participant_lookup:
                    participant = participant_lookup[speaker]
                    
                    # Update institute if needed
                    if institute_unknown and participant.get('affiliation'):
                        talk['Institute'] = participant['affiliation']
                        updated_institute += 1
                    
                    # Update country if needed
                    if country_unknown and participant.get('country'):
                        talk['Country'] = participant['country']
                        updated_country += 1
                else:
                    # Try fuzzy matching
                    for p_name, participant in participant_lookup.items():
                        # Check if either name contains the other
                        if speaker in p_name or p_name in speaker:
                            # Update institute if needed
                            if institute_unknown and participant.get('affiliation'):
                                talk['Institute'] = participant['affiliation']
                                updated_institute += 1
                            
                            # Update country if needed
                            if country_unknown and participant.get('country'):
                                talk['Country'] = participant['country']
                                updated_country += 1
                            
                            break
    
    print(f"Total speakers processed: {total_speakers}")
    print(f"Updated institute information: {updated_institute}")
    print(f"Updated country information: {updated_country}")
    
    return conference_data

# In the main function, replace the existing call to fix_unknown_institutes_from_participants with:
# Load participant data
participant_data = load_participant_data()

# Update speaker information from participant data
if participant_data:
    conference_data = update_speaker_info_from_participant_data(conference_data, participant_data)
else:
    print("No participant data available for updating speaker information")

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

