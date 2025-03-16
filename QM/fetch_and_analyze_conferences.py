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
    """Validate that the Indico URL is correct for the given year"""
    url = f"https://indico.cern.ch/export/event/{indico_id}.json?detail=contributions&pretty=yes"
    try:
        # Use GET and print the response details for debugging
        print(f"\nChecking URL: {url}")
        response = requests.get(url, timeout=10)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Print event title for verification
            event_data = data.get('results', [{}])[0]
            title = event_data.get('title', '')
            print(f"Event title: {title}")
            
            if f"QM{year}" in title or f"Quark Matter {year}" in title:
                return True, "Valid Indico page with correct year", data
            else:
                return False, f"Indico page exists but may not be for QM{year} (title: {title})", None
        else:
            return False, f"Indico page returns status code {response.status_code}", None
    except requests.exceptions.RequestException as e:
        return False, f"Error validating URL: {str(e)}", None

def categorize_session(session_name, year):
    """Categorize a session as plenary or parallel based on its name and year"""
    if not session_name:
        return "unknown"
        
    session_lower = str(session_name).lower()
    
    # Skip known non-talk sessions
    if any(x in session_lower for x in [
        'poster', 'flash', 'student', 'opening', 'closing',
        'welcome', 'teacher', 'public'
    ]):
        return "other"
    
    # Year-specific patterns
    if year == '2018':
        if 'plenary' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'jet modifications', 'collective dynamics', 'collectivity in small systems',
            'quarkonia', 'initial state physics', 'correlations and fluctuations',
            'open heavy flavour', 'chirality', 'phase diagram', 'qcd at high temperature',
            'electromagnetic and weak probes', 'new theoretical developments',
            'thermodynamics and hadron chemistry', 'high baryon density'
        ]):
            return "parallel"
            
    elif year == '2023':
        if 'plenary session' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'jets', 'heavy flavor', 'collective dynamics', 'new theory',
            'small systems', 'initial state', 'qcd at finite t',
            'light flavor', 'em probes', 'critical point', 'chirality',
            'spin/eic physics', 'future experiments', 'astrophysics', 'upc'
        ]):
            return "parallel"
            
    elif year == '2014':
        if 'plenary' in session_lower:
            return "plenary"
        if any(x in session_lower for x in [
            'heavy flavor', 'jets', 'correlations and fluctuations',
            'collective dynamics', 'qcd phase diagram', 'electromagnetic probes',
            'initial state physics', 'new theoretical developments',
            'thermodynamics and hadron chemistry', 'approach to equilibrium'
        ]):
            return "parallel"
            
    elif year == '2015':
        if 'plenary session' in session_lower:
            return "plenary"
        if any(x in session_lower.replace('-', ' ') for x in [
            'jets and high pt', 'correlations and fluctuations',
            'qgp in small systems', 'initial state physics',
            'open heavy flavors', 'collective dynamics',
            'quarkonia', 'electromagnetic probes'
        ]):
            return "parallel"
            
    else:
        # General patterns for other years
        if 'plenary' in session_lower:
            return "plenary"
        if 'parallel' in session_lower:
            return "parallel"
    
    return "unknown"

def fetch_and_process_contributions(indico_id, year):
    """Fetch and process contributions from Indico"""
    is_valid, message, data = validate_indico_url(indico_id, year)
    print(f"\nQM{year} (ID: {indico_id}): {message}")
    
    if not is_valid or not data:
        print(f"  Warning: Could not fetch valid data for QM{year}")
        return None
    
    try:
        results = data.get('results', [])
        if not results:
            return None
            
        event_data = results[0]
        contributions = event_data.get('contributions', [])
        
        # Print session distribution first
        session_counts = Counter()
        for contribution in contributions:
            session = contribution.get('session', '')
            session_counts[session] += 1
        
        print(f"\nQM{year} Session distribution:")
        for session, count in session_counts.most_common():
            print(f"  {session}: {count}")
            
        all_talks = []
        plenary_talks = []
        parallel_talks = []
        
        # Process contributions
        for contribution in contributions:
            session = contribution.get('session', '')
            session_type = categorize_session(session, year)
            
            # Basic talk data
            talk_data = {
                'Session': session,
                'Type': session_type,
                'Title': contribution.get('title', 'No title')
            }
            
            all_talks.append(talk_data)
            if session_type == "plenary":
                plenary_talks.append(talk_data)
            elif session_type == "parallel":
                parallel_talks.append(talk_data)
        
        print(f"\nQM{year} Categorization results:")
        print(f"  Total talks: {len(all_talks)}")
        print(f"  Plenary talks: {len(plenary_talks)}")
        print(f"  Parallel talks: {len(parallel_talks)}")
        print(f"  Uncategorized: {len(all_talks) - len(plenary_talks) - len(parallel_talks)}")
        
        return {
            'all_talks': all_talks,
            'plenary_talks': plenary_talks,
            'parallel_talks': parallel_talks
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
                        'metadata': data['metadata']
                    }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  Could not load processed data: {e}")
                continue
        
        # Print final summary table
        print("\nYear Location Total Plenary Parallel")
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
                print(f"{year} {location} {total} {plenary} {parallel}")
        
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
    # First, fetch and process all conferences
    print("=== FETCHING AND PROCESSING CONFERENCES ===")
    try:
        with open('listofQMindigo', 'r') as f:
            # Skip lines that start with '#' and take only first two items (year and ID)
            conferences = [line.strip().split()[:2] for line in f if not line.strip().startswith('#')]
            
        conference_data = {}
        processed_conferences = []
        
        # Process each conference fresh (no caching)
        for year, indico_id in conferences:
            print(f"\nProcessing QM{year} (ID: {indico_id})")
            conference_stats = fetch_and_process_contributions(indico_id, year)
            if conference_stats:
                conference_data[year] = conference_stats
                
        # Print final summary table
        print("\nYear Location Total Plenary Parallel")
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
                print(f"{year} {location} {total} {plenary} {parallel}")
                
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        exit(1) 