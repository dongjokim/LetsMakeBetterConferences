import requests
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Set font that supports CJK characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Create figs directory if it doesn't exist
os.makedirs('figs', exist_ok=True)

# Configuration
EVENT_ID = 1334113

# API endpoint for event contributions (public access)
url = f"https://indico.cern.ch/export/event/{EVENT_ID}.json?detail=contributions&pretty=yes"

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

def plot_distributions(all_data, plenary_data, parallel_data):
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: All talks per country
    plt.subplot(3, 2, 1)
    country_counts = Counter([talk['Country'] for talk in all_data])
    plt.bar(country_counts.keys(), country_counts.values())
    plt.title('All Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 2: All talks per institute (top 20)
    plt.subplot(3, 2, 2)
    institute_counts = Counter([talk['Institute'] for talk in all_data])
    top_20_institutes = dict(sorted(institute_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.bar(top_20_institutes.keys(), top_20_institutes.values())
    plt.title('All Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Plenary talks per country
    plt.subplot(3, 2, 3)
    plenary_country_counts = Counter([talk['Country'] for talk in plenary_data])
    plt.bar(plenary_country_counts.keys(), plenary_country_counts.values())
    plt.title('Plenary Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: Plenary talks per institute
    plt.subplot(3, 2, 4)
    plenary_institute_counts = Counter([talk['Institute'] for talk in plenary_data])
    plt.bar(plenary_institute_counts.keys(), plenary_institute_counts.values())
    plt.title('Plenary Talks by Institute')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 5: Parallel talks per country
    plt.subplot(3, 2, 5)
    parallel_country_counts = Counter([talk['Country'] for talk in parallel_data])
    plt.bar(parallel_country_counts.keys(), parallel_country_counts.values())
    plt.title('Parallel Talks by Country')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 6: Parallel talks per institute (top 20)
    plt.subplot(3, 2, 6)
    parallel_institute_counts = Counter([talk['Institute'] for talk in parallel_data])
    top_20_parallel = dict(sorted(parallel_institute_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.bar(top_20_parallel.keys(), top_20_parallel.values())
    plt.title('Parallel Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('figs/talk_distributions.pdf', bbox_inches='tight')
    
    # Create individual plots for better visibility
    # All talks country distribution
    plt.figure(figsize=(15, 8))
    plt.bar(country_counts.keys(), country_counts.values())
    plt.title('All Talks by Country')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/all_talks_country.pdf', bbox_inches='tight')
    
    # All talks institute distribution (top 20)
    plt.figure(figsize=(15, 8))
    plt.bar(top_20_institutes.keys(), top_20_institutes.values())
    plt.title('All Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/all_talks_institute.pdf', bbox_inches='tight')
    
    # Plenary talks country distribution
    plt.figure(figsize=(15, 8))
    plt.bar(plenary_country_counts.keys(), plenary_country_counts.values())
    plt.title('Plenary Talks by Country')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/plenary_talks_country.pdf', bbox_inches='tight')
    
    # Plenary talks institute distribution
    plt.figure(figsize=(15, 8))
    plt.bar(plenary_institute_counts.keys(), plenary_institute_counts.values())
    plt.title('Plenary Talks by Institute')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/plenary_talks_institute.pdf', bbox_inches='tight')
    
    # Parallel talks country distribution
    plt.figure(figsize=(15, 8))
    plt.bar(parallel_country_counts.keys(), parallel_country_counts.values())
    plt.title('Parallel Talks by Country')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/parallel_talks_country.pdf', bbox_inches='tight')
    
    # Parallel talks institute distribution (top 20)
    plt.figure(figsize=(15, 8))
    plt.bar(top_20_parallel.keys(), top_20_parallel.values())
    plt.title('Parallel Talks by Institute (Top 20)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/parallel_talks_institute.pdf', bbox_inches='tight')
    
    # Print statistics
    print("\nTotal number of talks:", len(all_data))
    print("\nTop 10 countries by number of talks:")
    for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{country}: {count}")
    print("\nTop 10 institutes by number of talks:")
    for inst, count in sorted(institute_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{inst}: {count}")

def fetch_and_process_contributions():
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No results found in the response")
                return
            
            all_talks = []
            plenary_talks = []
            parallel_talks = []
            
            # Process each contribution
            for contribution in results[0].get('contributions', []):
                title = contribution.get('title', 'No title')
                session = contribution.get('session', 'No session')
                
                # Try different possible paths for speaker information
                speakers = (contribution.get('speakers', []) or 
                          contribution.get('person_links', []) or 
                          contribution.get('primary_authors', []))
                
                # Process each speaker
                for speaker in speakers:
                    name = (speaker.get('name', '') or 
                           speaker.get('full_name', '') or 
                           f"{speaker.get('first_name', '')} {speaker.get('last_name', '')}")
                    
                    affiliation = (speaker.get('affiliation', '') or 
                                 speaker.get('institution', '') or 
                                 speaker.get('institute', ''))
                    
                    country = extract_country(affiliation)
                    
                    talk_data = {
                        'Session': session,
                        'Title': title,
                        'Speaker': name.strip() or 'No name',
                        'Institute': affiliation or 'No affiliation',
                        'Country': country
                    }
                    
                    all_talks.append(talk_data)
                    if 'Plenary' in str(session):
                        plenary_talks.append(talk_data)
                    elif 'Parallel' in str(session):
                        parallel_talks.append(talk_data)
            
            # Create plots and print statistics
            plot_distributions(all_talks, plenary_talks, parallel_talks)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    print("Fetching contribution details...")
    fetch_and_process_contributions() 

    