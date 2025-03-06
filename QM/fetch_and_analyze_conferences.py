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
    url = f"https://indico.cern.ch/event/{indico_id}/"
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            # Make a GET request to check the title
            response = requests.get(url, timeout=10)
            # Look for QM and the year in the page title
            if f"QM{year}" in response.text or f"Quark Matter {year}" in response.text:
                return True, "Valid Indico page with correct year"
            else:
                return False, f"Indico page exists but may not be for QM{year}"
        else:
            return False, f"Indico page returns status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Error validating URL: {str(e)}"

def fetch_and_process_contributions(indico_id, year):
    # Use the detailed API endpoint to get contributions
    url = f"https://indico.cern.ch/export/event/{indico_id}.json?detail=contributions&pretty=yes"
    
    # First validate the Indico URL
    is_valid, message = validate_indico_url(indico_id, year)
    print(f"QM{year} (ID: {indico_id}): {message}")
    
    if not is_valid:
        print(f"  Warning: Indico ID {indico_id} may not be correct for QM{year}")
    
    try:
        print(f"  Fetching contribution details...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print(f"  No results found for QM{year}")
                return None
            
            # Extract event metadata if available
            event_data = results[0]
            
            # Try different possible paths for location information
            conference_location = (
                event_data.get('location', '') or 
                event_data.get('room', '') or 
                event_data.get('venue', '') or
                event_data.get('address', '')
            )
            
            # If still no location, try to extract from event title or description
            if not conference_location:
                title = event_data.get('title', '')
                description = event_data.get('description', '')
                
                # Common location patterns in titles: "QM2018 in Venice" or "Quark Matter 2018, Venice, Italy"
                location_match = re.search(r'(?:in|at|,)\s+([^,]+(?:,\s*[^,]+)?)', title + ' ' + description)
                if location_match:
                    conference_location = location_match.group(1).strip()
                else:
                    # Manually map known conference years to locations
                    location_map = {
                        '2011': 'Annecy, France',
                        '2012': 'Washington DC, USA',
                        '2014': 'Darmstadt, Germany',
                        '2015': 'Kobe, Japan',
                        '2017': 'Chicago, USA',
                        '2018': 'Venice, Italy',
                        '2019': 'Wuhan, China',
                        '2022': 'Krakow, Poland',
                        '2023': 'Houston, USA',
                        '2025': 'Brisbane, Australia'
                    }
                    conference_location = location_map.get(year, 'Unknown location')
            
            if not conference_location:
                conference_location = 'Unknown location'
                
            print(f"  Extracted location: {conference_location}")
            
            conference_dates = event_data.get('startDate', '') + ' to ' + event_data.get('endDate', '')
            
            all_talks = []
            plenary_talks = []
            parallel_talks = []
            
            # Process each contribution
            for contribution in results[0].get('contributions', []):
                title = contribution.get('title', 'No title')
                session = contribution.get('session', 'No session')
                
                # Try different possible paths for speaker information
                # Indico schema can vary between events
                speakers = (contribution.get('speakers', []) or 
                          contribution.get('person_links', []) or 
                          contribution.get('primary_authors', []))
                
                # If no speakers found, check if there's a nested structure
                if not speakers and 'persons' in contribution:
                    speakers = contribution['persons']
                
                # Process each speaker
                for speaker in speakers:
                    # Handle different name formats in Indico
                    name = (speaker.get('name', '') or 
                           speaker.get('full_name', '') or 
                           f"{speaker.get('first_name', '')} {speaker.get('last_name', '')}")
                    
                    # Handle different affiliation formats
                    affiliation = (speaker.get('affiliation', '') or 
                                 speaker.get('institution', '') or 
                                 speaker.get('institute', '') or
                                 speaker.get('affiliation_link', {}).get('name', ''))
                    
                    country = extract_country(affiliation)
                    
                    # Save additional data if available
                    abstract = contribution.get('description', '')
                    start_time = contribution.get('startDate', {})
                    duration = contribution.get('duration', '')
                    board_number = contribution.get('board_number', '')
                    
                    talk_data = {
                        'Session': session,
                        'Title': title,
                        'Speaker': name.strip() or 'No name',
                        'Institute': affiliation or 'No affiliation',
                        'Country': country,
                        'Abstract': abstract,
                        'Start_Time': start_time,
                        'Duration': duration,
                        'Board_Number': board_number
                    }
                    
                    all_talks.append(talk_data)
                    
                    # Categorize talks based on session name
                    if any(plenary_term in str(session).lower() for plenary_term in ['plenary', 'keynote']):
                        plenary_talks.append(talk_data)
                    elif any(parallel_term in str(session).lower() for parallel_term in ['parallel', 'concurrent']):
                        parallel_talks.append(talk_data)
                    # Check for poster sessions
                    elif 'poster' in str(session).lower():
                        # You could add a poster list here if needed
                        pass
            
            # Add more metadata about the conference
            processed_data = {
                'metadata': {
                    'year': year,
                    'indico_id': indico_id,
                    'download_date': datetime.now().isoformat(),
                    'conference_location': conference_location,
                    'conference_dates': conference_dates,
                    'total_talks': len(all_talks),
                    'plenary_talks': len(plenary_talks),
                    'parallel_talks': len(parallel_talks)
                },
                'all_talks': all_talks,
                'plenary_talks': plenary_talks,
                'parallel_talks': parallel_talks
            }
            
            # Save processed data
            with open(f'data/QM{year}_processed_data.json', 'w') as f:
                json.dump(processed_data, f, indent=2)
                
            print(f"  Successfully processed: {len(all_talks)} talks, {len(plenary_talks)} plenary, {len(parallel_talks)} parallel")
            print(f"  Location: {conference_location}, Dates: {conference_dates}")
            
            # Generate plots without detailed output
            return plot_distributions(all_talks, plenary_talks, parallel_talks, year, verbose=False)
            
        else:
            print(f"  Failed to fetch data: HTTP {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON response: {str(e)}")
        return None
    except Exception as e:
        print(f"  Error processing data: {str(e)}")
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
    # Read conference IDs
    try:
        with open('listofQMindigo', 'r') as f:
            # Skip lines that start with '#' and split the remaining lines
            conferences = [line.strip().split() for line in f if not line.strip().startswith('#')]
    except FileNotFoundError:
        print("Error: 'listofQMindigo' file not found")
        return
    
    print("\n=== VALIDATING INDICO PAGES FOR QM CONFERENCES ===")
    conference_data = {}
    processed_conferences = []
    
    for year, indico_id in conferences:
        print(f"\n=== QM{year} (ID: {indico_id}) ===")
        
        # Check if we already have processed data
        processed_file = f'data/QM{year}_processed_data.json'
        if os.path.exists(processed_file):
            print(f"  Processed data already exists, loading from file...")
            try:
                # Still validate the URL even if we have the data
                is_valid, message = validate_indico_url(indico_id, year)
                print(f"  {message}")
                
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    location = metadata.get('conference_location', 'Unknown location')
                    print(f"  Found location in metadata: {location}")
                    
                    all_talks = data.get('all_talks', [])
                    plenary_talks = data.get('plenary_talks', [])
                    parallel_talks = data.get('parallel_talks', [])
                    
                    # Store metadata for summary
                    processed_conferences.append({
                        'year': year,
                        'indico_id': indico_id,
                        'location': location,
                        'dates': metadata.get('conference_dates', ''),
                        'all_talks': len(all_talks),
                        'plenary_talks': len(plenary_talks),
                        'parallel_talks': len(parallel_talks),
                    })
                
                # Generate plots from existing data without verbose output
                conference_stats = plot_distributions(all_talks, plenary_talks, parallel_talks, year, verbose=False)
                if conference_stats:
                    conference_data[year] = conference_stats
                    print(f"  Summary: {len(all_talks)} talks, {len(plenary_talks)} plenary, {len(parallel_talks)} parallel")
            except Exception as e:
                print(f"  Error loading processed data: {e}")
                # Try to fetch and process again
                conference_stats = fetch_and_process_contributions(indico_id, year)
                if conference_stats:
                    conference_data[year] = conference_stats
        else:
            # Fetch and process data
            conference_stats = fetch_and_process_contributions(indico_id, year)
            if conference_stats:
                conference_data[year] = conference_stats
                
                # Read the newly created file to get metadata
                try:
                    with open(processed_file, 'r') as f:
                        data = json.load(f)
                        processed_conferences.append({
                            'year': year,
                            'indico_id': indico_id,
                            'location': data.get('metadata', {}).get('conference_location', 'Unknown'),
                            'dates': data.get('metadata', {}).get('conference_dates', ''),
                            'all_talks': conference_stats['all_talks'],
                            'plenary_talks': conference_stats['plenary_talks'],
                            'parallel_talks': conference_stats['parallel_talks'],
                        })
                except Exception as e:
                    print(f"  Error reading processed data for summary: {e}")
    
    # After processing all conferences, perform trend analysis
    if conference_data:
        print("\n=== GENERATING CROSS-CONFERENCE ANALYSIS ===")
        analyze_trends_across_conferences(conference_data)
        
        # Print final summary table with consistent data
        print("\n=== SUMMARY OF QM CONFERENCES ===")
        print(f"{'Year':<6} {'Location':<25} {'Total':<8} {'Plenary':<8} {'Parallel':<8}")
        print("-" * 65)
        
        # Sort by year for consistent display
        processed_conferences.sort(key=lambda x: x['year'])
        
        for conf in processed_conferences:
            year = conf['year']
            location = conf['location']
            if len(location) > 23:
                location = location[:20] + "..."
            total = conf['all_talks']
            plenary = conf['plenary_talks']
            parallel = conf['parallel_talks']
            
            print(f"{year:<6} {location:<25} {total:<8} {plenary:<8} {parallel:<8}")
            
        # Save summary to CSV for easy reference
        try:
            with open('data/conference_summary.csv', 'w') as f:
                f.write("Year,Location,Total Talks,Plenary Talks,Parallel Talks\n")
                for conf in processed_conferences:
                    f.write(f"{conf['year']},{conf['location']},{conf['all_talks']},{conf['plenary_talks']},{conf['parallel_talks']}\n")
            print("\nSummary saved to data/conference_summary.csv")
        except Exception as e:
            print(f"Error saving summary to CSV: {e}")
    else:
        print("No conference data was successfully processed for trend analysis")

if __name__ == "__main__":
    fetch_and_analyze_conferences() 