import requests
import json
import re
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def extract_country(affiliation):
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
    
    # Check against known institutions
    for inst, country in INSTITUTION_COUNTRY.items():
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
                
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 