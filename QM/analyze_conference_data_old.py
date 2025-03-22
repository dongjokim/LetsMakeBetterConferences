import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from collections import Counter

# Create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

def load_conference_data(file_path='data/final/all_conferences.json'):
    """
    Load conference data from JSON file.
    
    Parameters:
    - file_path: Path to the JSON file with conference data
    
    Returns:
    - Dictionary with conference data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded conference data from {file_path}")
            print(f"Data contains information for {len(data)} conferences")
            
            # Print more detailed information about the structure
            for year, year_data in data.items():
                print(f"\nYear {year} data:")
                if isinstance(year_data, dict):
                    for key in year_data:
                        if isinstance(year_data[key], list):
                            print(f"  {key}: {len(year_data[key])} items")
                        else:
                            print(f"  {key}: {type(year_data[key])}")
                else:
                    print(f"  Unexpected format: {type(year_data)}")
            
            return data
    except FileNotFoundError:
        print(f"Error: Conference data file not found: {file_path}")
        print("Please run 'python QM/generate_conference_data.py' first to generate the data.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in conference data file: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading conference data: {e}")
        return {}

def convert_to_dataframes(conference_data):
    """
    Convert conference data to pandas DataFrames for easier analysis.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Dictionary of DataFrames for different talk types
    """
    print("\nConverting conference data to DataFrames...")
    
    # Create empty lists to store data
    all_talks = []
    plenary_talks = []
    parallel_talks = []
    poster_talks = []
    
    # Process each conference
    for year, data in conference_data.items():
        # Process all talks
        for talk in data.get('all_talks', []):
            talk_dict = {
                'Year': year,
                'Title': talk.get('Title', ''),
                'Speaker': talk.get('Speaker', ''),
                'Institute': talk.get('Institute', ''),
                'Country': talk.get('Country', ''),
                'Date': talk.get('Date', ''),
                'TalkType': talk.get('TalkType', '')
            }
            all_talks.append(talk_dict)
        
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            talk_dict = {
                'Year': year,
                'Title': talk.get('Title', ''),
                'Speaker': talk.get('Speaker', ''),
                'Institute': talk.get('Institute', ''),
                'Country': talk.get('Country', ''),
                'Date': talk.get('Date', ''),
                'TalkType': 'Plenary'
            }
            plenary_talks.append(talk_dict)
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            talk_dict = {
                'Year': year,
                'Title': talk.get('Title', ''),
                'Speaker': talk.get('Speaker', ''),
                'Institute': talk.get('Institute', ''),
                'Country': talk.get('Country', ''),
                'Date': talk.get('Date', ''),
                'TalkType': 'Parallel'
            }
            parallel_talks.append(talk_dict)
        
        # Process poster talks
        for talk in data.get('poster_talks', []):
            talk_dict = {
                'Year': year,
                'Title': talk.get('Title', ''),
                'Speaker': talk.get('Speaker', ''),
                'Institute': talk.get('Institute', ''),
                'Country': talk.get('Country', ''),
                'Date': talk.get('Date', ''),
                'TalkType': 'Poster'
            }
            poster_talks.append(talk_dict)
    
    # Convert to DataFrames
    df_all = pd.DataFrame(all_talks)
    df_plenary = pd.DataFrame(plenary_talks)
    df_parallel = pd.DataFrame(parallel_talks)
    df_poster = pd.DataFrame(poster_talks)
    
    print(f"Created DataFrames with {len(df_all)} total talks:")
    print(f"  - {len(df_plenary)} plenary talks")
    print(f"  - {len(df_parallel)} parallel talks")
    print(f"  - {len(df_poster)} poster talks")
    
    return {
        'all': df_all,
        'plenary': df_plenary,
        'parallel': df_parallel,
        'poster': df_poster
    }

def analyze_country_distribution(conference_data):
    """
    Analyze the distribution of countries across all conferences.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Dictionary with country counts for different talk types
    """
    print("\nAnalyzing country distribution across all conferences...")
    
    # Create counters for different talk types
    all_countries = Counter()
    plenary_countries = Counter()
    parallel_countries = Counter()
    poster_countries = Counter()
    
    # Process each conference
    for year, data in conference_data.items():
        # Process all talks
        for talk in data.get('all_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                all_countries[country] += 1
        
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                plenary_countries[country] += 1
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                parallel_countries[country] += 1
        
        # Process poster talks
        for talk in data.get('poster_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                poster_countries[country] += 1
    
    # Print top countries overall
    print("\nTop 10 countries across all talk types:")
    for country, count in all_countries.most_common(10):
        percentage = (count / sum(all_countries.values())) * 100
        print(f"  {country}: {count} talks ({percentage:.1f}%)")
    
    # Print top countries for plenary talks
    print("\nTop 10 countries for plenary talks:")
    for country, count in plenary_countries.most_common(10):
        percentage = (count / sum(plenary_countries.values())) * 100
        print(f"  {country}: {count} talks ({percentage:.1f}%)")
    
    return {
        'all': all_countries,
        'plenary': plenary_countries,
        'parallel': parallel_countries,
        'poster': poster_countries
    }

def analyze_plenary_vs_parallel(conference_data):
    """
    Analyze the representation ratio between plenary and parallel talks by country.
    
    Parameters:
    - conference_data: Dictionary with conference data
    
    Returns:
    - Tuple of (plenary_country, parallel_country) counters
    """
    print("\nAnalyzing representation between plenary and parallel talks...")
    
    # Create counters for plenary and parallel talks
    plenary_country = Counter()
    parallel_country = Counter()
    
    # Process each conference
    for year, data in conference_data.items():
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                plenary_country[country] += 1
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            country = talk.get('Country', '')
            if country and country != 'Unknown':
                parallel_country[country] += 1
    
    # Calculate representation ratios
    plenary_total = sum(plenary_country.values())
    parallel_total = sum(parallel_country.values())
    
    print("\nRepresentation ratios (plenary share / parallel share):")
    ratios = []
    for country in set(plenary_country) | set(parallel_country):
        if country in plenary_country and country in parallel_country:
            plenary_share = plenary_country[country] / plenary_total
            parallel_share = parallel_country[country] / parallel_total
            ratio = plenary_share / parallel_share if parallel_share > 0 else float('inf')
            ratios.append((country, ratio))
    
    # Sort by ratio in descending order
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Print top and bottom 5 ratios
    print("\nTop 5 overrepresented countries in plenary talks:")
    for country, ratio in ratios[:5]:
        print(f"  {country}: {ratio:.2f}x")
    
    print("\nBottom 5 underrepresented countries in plenary talks:")
    for country, ratio in ratios[-5:]:
        print(f"  {country}: {ratio:.2f}x")
    
    return plenary_country, parallel_country

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
        
        if parallel_share > 0:  # Avoid division by zero
            ratio = plenary_share / parallel_share
            ratios.append((country, ratio))
    
    # Sort by ratio
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Get top and bottom countries
    top_10 = ratios[:10]  # Top 10 overrepresented
    
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
    Create plots showing regional diversity across conferences.
    
    Parameters:
    - country_counts: Dictionary with country counts for different talk types
    - conference_data: Dictionary with conference data
    """
    print("\nCreating regional diversity plots...")
    
    # Define regions and their countries
    regions = {
        'North America': ['USA', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Netherlands', 
                 'Spain', 'Poland', 'Finland', 'Denmark', 'Sweden', 'Norway', 
                 'Belgium', 'Austria', 'Hungary', 'Czech Republic', 'Portugal',
                 'Greece', 'Ireland', 'Russia', 'Ukraine', 'Romania', 'Croatia'],
        'Asia': ['Japan', 'China', 'India', 'Korea', 'Taiwan', 'Singapore', 
               'Israel', 'Turkey', 'Iran', 'Pakistan'],
        'Other': ['Brazil', 'Australia', 'South Africa', 'Argentina', 'Chile', 
                'New Zealand', 'Egypt', 'South Africa', 'Mexico', 'Colombia']
    }
    
    # Rest of function...

def load_conference_dataframes(base_dir="data/final"):
    """
    Load the saved pandas DataFrames from pickle files.
    
    Parameters:
    - base_dir: Directory containing the saved DataFrames
    
    Returns:
    - Dictionary of DataFrames for different talk types
    """
    dataframes = {}
    try:
        # Types of DataFrames to load
        types = ['all', 'plenary', 'parallel', 'poster']
        
        for df_type in types:
            pickle_path = f"{base_dir}/{df_type}_talks.pkl"
            try:
                df = pd.read_pickle(pickle_path)
                dataframes[df_type] = df
                print(f"Loaded {df_type} talks DataFrame with {len(df)} records")
            except FileNotFoundError:
                print(f"Warning: DataFrame file {pickle_path} not found")
        
        return dataframes
    except Exception as e:
        print(f"Error loading DataFrames: {e}")
        return {}

def create_keywords_plot(conference_data):
    """
    Create a plot showing the most common keywords across all conferences.
    
    Parameters:
    - conference_data: Dictionary with conference data
    """
    print("\nAnalyzing keywords across conferences...")
    
    # Initialize keyword counter
    all_keywords = Counter()
    
    # Process each conference
    for year, data in conference_data.items():
        # Process all talks
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            for talk in data.get(talk_type, []):
                title = talk.get('Title', '')
                if title:
                    # Extract keywords from title
                    keywords = extract_keywords(title)
                    all_keywords.update(keywords)
    
    # Get top keywords
    top_keywords = all_keywords.most_common(20)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Extract names and counts
    names = [item[0] for item in top_keywords]
    values = [item[1] for item in top_keywords]
    
    # Create horizontal bars
    y_pos = np.arange(len(names))
    plt.barh(y_pos, values, align='center')
    
    plt.yticks(y_pos, names)
    plt.xlabel('Frequency')
    plt.title('Most Common Keywords in Talk Titles')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/keywords_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Print the top keywords
    print("\nTop 20 Keywords:")
    for keyword, count in top_keywords:
        print(f"  {keyword}: {count}")

def extract_keywords(title):
    """
    Extract meaningful keywords from a talk title.
    
    Parameters:
    - title: The talk title
    
    Returns:
    - List of keywords
    """
    # Convert to lowercase
    title = title.lower()
    
    # Remove punctuation
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Split into words
    words = title.split()
    
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'with', 'from', 'by', 'about', 'as', 'of', 'over', 'between',
                 'through', 'after', 'before', 'during', 'qm', 'quark', 'matter',
                 'results', 'measurement', 'measurements', 'study', 'studies',
                 'using', 'new', 'recent', 'first', 'latest'}
    
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

def create_institute_bubble_chart(conference_data):
    """
    Create a bubble chart showing the distribution of talks by institute across different talk types.
    
    Parameters:
    - conference_data: Dictionary with conference data
    """
    print("\nCreating institute bubble chart...")
    
    # Count all institutes by talk type
    all_institutes = Counter()
    plenary_institutes = Counter()
    parallel_institutes = Counter()
    poster_institutes = Counter()
    
    # Process each conference
    for year, data in conference_data.items():
        # Process plenary talks
        for talk in data.get('plenary_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute != 'Unknown':
                plenary_institutes[institute] += 1
                all_institutes[institute] += 1
        
        # Process parallel talks
        for talk in data.get('parallel_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute != 'Unknown':
                parallel_institutes[institute] += 1
                all_institutes[institute] += 1
        
        # Process poster talks
        for talk in data.get('poster_talks', []):
            institute = talk.get('Institute', '')
            if institute and institute != 'Unknown':
                poster_institutes[institute] += 1
                all_institutes[institute] += 1
    
    # Get top 30 institutes
    top_institutes = [inst for inst, _ in all_institutes.most_common(30)]
    
    # Prepare data for bubble chart
    institute_data = []
    for institute in top_institutes:
        total = all_institutes[institute]
        plenary = plenary_institutes[institute]
        parallel = parallel_institutes[institute]
        poster = poster_institutes[institute]
        
        # Calculate plenary-to-parallel ratio (with handling for zero division)
        ratio = plenary / parallel if parallel > 0 else 0
        
        institute_data.append({
            'Institute': institute,
            'Total': total,
            'Plenary': plenary,
            'Parallel': parallel,
            'Poster': poster,
            'Ratio': ratio
        })
    
    # Create a DataFrame
    df = pd.DataFrame(institute_data)
    
    # Create the bubble chart
    plt.figure(figsize=(14, 10))
    
    # Define color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    # Create scatter plot
    scatter = plt.scatter(
        df['Parallel'], 
        df['Plenary'],
        s=df['Total'] * 10,  # Size based on total talks
        c=colors,
        alpha=0.6,
        edgecolors='w'
    )
    
    # Add labels for each institute
    for i, row in df.iterrows():
        plt.annotate(
            row['Institute'],
            xy=(row['Parallel'], row['Plenary']),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=9
        )
    
    # Add diagonal line for equal representation
    max_val = max(df['Parallel'].max(), df['Plenary'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Number of Parallel Talks')
    plt.ylabel('Number of Plenary Talks')
    plt.title('Institute Representation: Plenary vs. Parallel Talks')
    plt.grid(alpha=0.3)
    
    # Save the figure
    plt.savefig('figures/institute_bubble_chart.pdf', bbox_inches='tight')
    plt.savefig('figures/institute_bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Institute bubble chart created successfully.")

def main():
    """
    Main function to load data and generate all visualizations.
    """
    print("QM Conference Analysis")
    print("=====================")
    
    # Check if conference data exists and is populated
    conference_data = load_conference_data()
    
    # If data is empty, try to fetch it directly
    if not conference_data or all(not data.get('all_talks') for data in conference_data.values()):
        print("Empty or missing conference data. Trying to fetch and process data directly...")
        
        try:
            # This is the approach from fetch_and_analyze_conferences.py that worked
            indico_ids = load_indico_ids_from_file('listofQMindigo')
            
            conference_data = {}
            for year, indico_id in sorted(indico_ids.items()):
                print(f"\nProcessing QM{year} (Indico ID: {indico_id})...")
                # Directly call fetch_and_process_contributions from generate_conference_data.py
                data = fetch_and_process_contributions(indico_id, year)
                if data:
                    conference_data[year] = data
            
            # Process and update the data
            participant_data = load_participant_data()
            if participant_data:
                conference_data = update_speaker_info_from_participant_data(conference_data, participant_data)
            
            # Save the fetched and processed data
            print("\nSaving processed conference data...")
            os.makedirs('data/final', exist_ok=True)
            
            with open('data/final/all_conferences.json', 'w', encoding='utf-8') as f:
                json.dump(conference_data, f, indent=2)
                print("Saved all conference data to data/final/all_conferences.json")
            
            # Save individual conference data files
            for year, data in conference_data.items():
                output_file = f"data/processed/qm{year}_processed.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved processed data for QM{year} to {output_file}")
                except Exception as e:
                    print(f"Error saving processed data for QM{year}: {e}")
        
        except Exception as e:
            print(f"Error fetching and processing data: {e}")
            print("Please run 'python QM/generate_conference_data.py' with correct parameters.")
            return
    
    # Convert to DataFrames for easier analysis
    dataframes = convert_to_dataframes(conference_data)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # Country analysis
    country_counts = analyze_country_distribution(conference_data)
    create_plenary_country_plot(country_counts['plenary'])
    
    # Representation ratio analysis
    plenary_country, parallel_country = analyze_plenary_vs_parallel(conference_data)
    create_representation_ratio_plot(plenary_country, parallel_country)
    
    # Regional diversity analysis
    create_regional_diversity_plot(country_counts, conference_data)
    
    # Keywords analysis
    create_keywords_plot(conference_data)
    
    # Institute analysis
    create_institute_bubble_chart(conference_data)
    
    print("\nAll visualizations have been generated and saved to the 'figures' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main()