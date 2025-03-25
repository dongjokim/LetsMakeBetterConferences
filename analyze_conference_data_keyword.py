#!/usr/bin/env python3
"""
analyze_conference_data_keyword.py - Simplified version focused on keyword analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import os
import json
import traceback
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Only keep the font settings needed for plots
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 'large',
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
os.makedirs('figures', exist_ok=True)

# Conference locations for reference
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
    '2025': 'Frankfurt, Germany'
}

def load_processed_data(base_dir='data/processed'):
    """Load processed conference data from CSV files"""
    conference_data = {}
    
    try:
        # Check if base directory exists
        if not os.path.exists(base_dir):
            print(f"Error: Processed data directory not found at {base_dir}")
            print("Please run generate_conference_data.py first")
            return None
            
        # Get all year directories
        year_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        for year in sorted(year_dirs):
            year_path = os.path.join(base_dir, year)
            year_data = {}
            
            # Load each talk type
            for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks', 'all_talks']:
                file_path = os.path.join(year_path, f'{talk_type}.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    year_data[talk_type] = df.to_dict('records')
            
            # Load statistics if they exist
            stats_file = os.path.join(year_path, 'statistics.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    year_data.update(stats)
            
            conference_data[year] = year_data
            
        print(f"\nLoaded processed data for {len(conference_data)} conferences")
        return conference_data
        
    except Exception as e:
        print(f"Error loading processed data: {e}")
        traceback.print_exc()
        return None

def display_conference_summary(conference_data):
    """Display a summary of the conference data"""
    print("\nConference Summary:")
    print("===================")
    
    for year in sorted(conference_data.keys()):
        data = conference_data[year]
        
        # Get counts
        plenary_count = len(data.get('plenary_talks', []))
        parallel_count = len(data.get('parallel_talks', []))
        poster_count = len(data.get('poster_talks', []))
        
        # Get country stats
        country_counts = Counter()
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            for talk in data.get(talk_type, []):
                country = talk.get('Country', 'Unknown')
                country_counts[country] += 1
        
        # Display summary for this year
        print(f"\nQM {year} - {CONFERENCE_LOCATIONS.get(year, 'Location unknown')}")
        print(f"  Plenary talks: {plenary_count}")
        print(f"  Parallel talks: {parallel_count}")
        print(f"  Poster talks: {poster_count}")
        print(f"  Total contributions: {plenary_count + parallel_count + poster_count}")
        
        # Show top countries
        print("  Top 5 countries:")
        for country, count in country_counts.most_common(5):
            print(f"    {country}: {count} ({count/(plenary_count + parallel_count + poster_count)*100:.1f}%)")

def fix_unknown_countries_for_known_institutes(conference_data):
    """Fix unknown countries for known institutes"""
    # Create a mapping of institutes to countries
    institute_to_country = {}
    
    # First, collect all known institute->country mappings
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Institute' in talk and 'Country' in talk:
                        institute = talk['Institute']
                        country = talk['Country']
                        if institute != 'Unknown' and country != 'Unknown':
                            institute_to_country[institute] = country
    
    # Now fix unknown countries
    updated_count = 0
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Institute' in talk and 'Country' in talk:
                        if talk['Country'] == 'Unknown' and talk['Institute'] in institute_to_country:
                            talk['Country'] = institute_to_country[talk['Institute']]
                            updated_count += 1
    
    print(f"Fixed {updated_count} unknown countries using known institute mappings")
    return conference_data

def fix_unknown_institute_country_data(conference_data):
    """Fix unknown institute and country data"""
    # Create mappings of names to institutes and countries
    name_to_institute = {}
    name_to_country = {}
    
    # Collect known mappings
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Speaker' in talk and 'Institute' in talk and 'Country' in talk:
                        name = talk['Speaker']
                        institute = talk['Institute']
                        country = talk['Country']
                        
                        if name and institute != 'Unknown':
                            name_to_institute[name] = institute
                        
                        if name and country != 'Unknown':
                            name_to_country[name] = country
    
    # Fix unknown data
    updated_institute_count = 0
    updated_country_count = 0
    
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Speaker' in talk:
                        name = talk['Speaker']
                        
                        # Fix unknown institute
                        if talk.get('Institute') == 'Unknown' and name in name_to_institute:
                            talk['Institute'] = name_to_institute[name]
                            updated_institute_count += 1
                        
                        # Fix unknown country
                        if talk.get('Country') == 'Unknown' and name in name_to_country:
                            talk['Country'] = name_to_country[name]
                            updated_country_count += 1
    
    print(f"Fixed {updated_institute_count} unknown institutes and {updated_country_count} unknown countries")
    return conference_data

def fix_common_affiliation_problems(conference_data):
    """Fix common problems with affiliations"""
    # Common fixes for country names
    country_fixes = {
        'United States': 'USA',
        'United States of America': 'USA',
        'U.S.A.': 'USA',
        'U.S.': 'USA',
        'US': 'USA',
        'united states': 'USA',
        'usa': 'USA',
        'United Kingdom': 'UK',
        'Great Britain': 'UK',
        'People\'s Republic of China': 'China',
        'P.R. China': 'China',
        'PR China': 'China',
        'P.R.China': 'China',
        'Republic of Korea': 'South Korea',
        'Korea': 'South Korea',
        'Russian Federation': 'Russia',
        'Deutschland': 'Germany',
        'Czech Republik': 'Czech Republic',
        'Brasil': 'Brazil',
        'Suisse': 'Switzerland'
    }
    
    updated_count = 0
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Country' in talk and talk['Country'] in country_fixes:
                        talk['Country'] = country_fixes[talk['Country']]
                        updated_count += 1
    
    print(f"Fixed {updated_count} common country naming issues")
    return conference_data

def add_manual_country_fixes(conference_data):
    """Add manual fixes for specific institutes to countries"""
    institute_to_country = {
        'Harvard University': 'USA',
        'MIT': 'USA',
        'University of California, Berkeley': 'USA',
        'Berkeley Lab': 'USA',
        'LBNL': 'USA',
        'Lawrence Berkeley National Laboratory': 'USA',
        'CERN': 'Switzerland',
        'University of Tokyo': 'Japan',
        'Tsinghua University': 'China',
        'Peking University': 'China',
        'University of Oxford': 'UK',
        'University of Cambridge': 'UK',
        'Imperial College London': 'UK',
        'Heidelberg University': 'Germany',
        'University of Heidelberg': 'Germany',
        'Université Paris-Saclay': 'France',
        'INFN': 'Italy',
        'JINR': 'Russia'
    }
    
    updated_count = 0
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if 'Institute' in talk and talk['Institute'] in institute_to_country:
                        # Only update if country is Unknown or if institute is a major international org
                        if talk.get('Country') == 'Unknown' or talk['Institute'] in ['CERN', 'JINR', 'INFN']:
                            talk['Country'] = institute_to_country[talk['Institute']]
                            updated_count += 1
    
    print(f"Added {updated_count} manual country fixes")
    return conference_data

def fix_unknown_institutes(conference_data):
    """Fix institutes marked as Unknown"""
    # For talks with unknown institutes but known countries, set institute to "Unknown [Country]"
    updated_count = 0
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    if talk.get('Institute') == 'Unknown' and talk.get('Country') != 'Unknown':
                        talk['Institute'] = f"Unknown institute ({talk['Country']})"
                        updated_count += 1
    
    print(f"Fixed {updated_count} unknown institutes by adding country information")
    return conference_data

def filter_relevant_talk_types(conference_data):
    """Filter to only include relevant talk types"""
    filtered_data = {}
    
    for year, data in conference_data.items():
        filtered_data[year] = {}
        
        # Copy relevant talk types
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                filtered_data[year][talk_type] = data[talk_type]
        
        # Copy statistics
        for key, value in data.items():
            if key not in ['plenary_talks', 'parallel_talks', 'poster_talks', 'all_talks']:
                filtered_data[year][key] = value
    
    return filtered_data

def preprocess_conference_data(conference_data):
    """Preprocess conference data to ensure consistent structure"""
    preprocessed_data = {}
    
    for year in conference_data:
        # Skip keys that aren't years
        if not str(year).isdigit():
            continue
        
        year_data = {}
        
        # Ensure each talk type exists
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                # Make a copy to avoid modifying the original
                talks = []
                for talk in conference_data[year][talk_type]:
                    # Create a clean talk object with standard fields
                    clean_talk = {
                        'Session': talk.get('Session', ''),
                        'Type': talk.get('Type', ''),
                        'Title': talk.get('Title', ''),
                        'Speaker': talk.get('Speaker', ''),
                        'Institute': talk.get('Institute', 'Unknown'),
                        'Country': talk.get('Country', 'Unknown'),
                        'Abstract': talk.get('Abstract', '')
                    }
                    talks.append(clean_talk)
                year_data[talk_type] = talks
            else:
                year_data[talk_type] = []
        
        # Copy any statistics or additional data
        for key, value in conference_data[year].items():
            if key not in ['plenary_talks', 'parallel_talks', 'poster_talks', 'all_talks']:
                year_data[key] = value
        
        preprocessed_data[year] = year_data
    
    return preprocessed_data

def extract_keywords_from_talk(talk):
    """Extract keywords from a talk"""
    keywords = []
    
    # Check if the talk has a 'Title' field
    title = talk.get('Title', '')
    if not title:
        return keywords
    
    # Try to extract keywords using common patterns
    # Pattern 1: Keywords: keyword1, keyword2, keyword3
    match = re.search(r'Keywords?:?\s*([^\.]+)', title, re.IGNORECASE)
    if match:
        keyword_text = match.group(1).strip()
        # Split by commas or semicolons
        for kw in re.split(r'[,;]', keyword_text):
            keyword = kw.strip().lower()
            if keyword:
                keywords.append(keyword)
    
    return keywords

def analyze_keywords(conference_data):
    """Analyze keywords in conference data and generate visualizations"""
    print("Analyzing keywords across conferences...")
    
    # Define keyword groups for tracking
    keyword_groups = {
        'QCD Matter': ['QGP', 'quark', 'gluon', 'plasma', 'deconfinement'],
        'Heavy-Ion Collisions': ['heavy-ion', 'nucleus-nucleus', 'collisions', 'RHIC', 'LHC'],
        'Phase Transitions': ['phase', 'transition', 'critical', 'crossover', 'first-order'],
        'Flow Phenomena': ['flow', 'collective', 'hydrodynamics', 'viscosity', 'anisotropic'],
        'Jets & Hard Probes': ['jet', 'quenching', 'fragmentation', 'hard', 'suppression'],
        'Small Systems': ['small', 'pp', 'p-p', 'proton-proton', 'pA'],
        'EoS & Transport': ['equation', 'state', 'transport', 'medium', 'properties']
    }
    
    # Get all years
    years = []
    for year in conference_data.keys():
        if str(year).isdigit():
            years.append(str(year))
    years.sort(key=int)  # Sort numerically
    
    # Define markers, colors, and line styles for plotting
    all_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h']
    all_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    all_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Create figure with panels for each keyword group
    fig, axs = plt.subplots(len(keyword_groups), 1, figsize=(12, 4 * len(keyword_groups)))
    
    # Analyze each keyword group
    for i, (group, keywords) in enumerate(keyword_groups.items()):
        ax = axs[i]
        
        # For each keyword in the group, track separately
        for j, keyword in enumerate(keywords[:5]):  # Limit to first 5 keywords to avoid overcrowding
            keyword_trend = []
            
            for year in years:
                # Get all talk titles for this year
                titles = []
                for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
                    if talk_type in conference_data[year]:
                        titles.extend([talk.get('Title', '') for talk in conference_data[year][talk_type]])
                
                # Convert to lowercase for case-insensitive matching
                titles = [title.lower() for title in titles if title]
                total_talks = len(titles)
                
                # Count keyword occurrences
                keyword_count = sum(1 for title in titles if keyword in title)
                
                # Store as percentage
                if total_talks > 0:
                    keyword_trend.append(keyword_count / total_talks * 100)
                else:
                    keyword_trend.append(0)
            
            # Use different marker, color, and linestyle for each keyword
            marker = all_markers[j % len(all_markers)]
            color = all_colors[j % len(all_colors)]
            linestyle = all_linestyles[j % len(all_linestyles)]
            
            # Plot this keyword's trend
            ax.plot(years, keyword_trend, marker=marker, linestyle=linestyle, 
                    linewidth=2, color=color, markersize=8, label=keyword)
        
        # Add legend inside each panel
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Set title and labels
        ax.set_title(group)
        ax.set_ylabel('% of Talks')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-label
    fig.text(0.5, 0.04, 'Conference Year', ha='center', fontsize=14)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig('figures/QM_keyword_analysis_basic.pdf')
    plt.close()
    
    return keyword_groups

def create_keyword_QA_plots(conference_data):
    """Create quality assurance plots for keywords analysis"""
    print("Creating keyword QA visualization...")
    
    # Get all years
    years = []
    for year in conference_data.keys():
        if str(year).isdigit():
            years.append(str(year))
    years.sort(key=int)  # Sort numerically
    
    # Create figure with multiple panels
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Panel 1: Number of talks with keywords over time
    ax = axs[0]
    talks_with_keywords = []
    total_talks = []
    
    for year in years:
        # Get all talks for this year
        year_talks = []
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                year_talks.extend(conference_data[year][talk_type])
        
        total_talks.append(len(year_talks))
        
        # Count talks with keywords
        talks_with_kw = 0
        for talk in year_talks:
            keywords = extract_keywords_from_talk(talk)
            if keywords:
                talks_with_kw += 1
        
        talks_with_keywords.append(talks_with_kw)
    
    # Calculate percentage
    keyword_percentages = [100 * kw / total if total > 0 else 0 
                          for kw, total in zip(talks_with_keywords, total_talks)]
    
    # Plot
    ax.bar(years, keyword_percentages, color='skyblue')
    ax.set_title('% of Talks with Keywords')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Conference Year')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(keyword_percentages):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Panel 2: Average number of keywords per talk
    ax = axs[1]
    avg_keywords = []
    
    for year in years:
        # Get all talks for this year
        year_talks = []
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                year_talks.extend(conference_data[year][talk_type])
        
        # Calculate average number of keywords
        if year_talks:
            total_keywords = sum(len(extract_keywords_from_talk(talk)) for talk in year_talks)
            avg = total_keywords / len(year_talks)
        else:
            avg = 0
        
        avg_keywords.append(avg)
    
    # Plot
    ax.bar(years, avg_keywords, color='lightgreen')
    ax.set_title('Average Keywords per Talk')
    ax.set_ylabel('Number of Keywords')
    ax.set_xlabel('Conference Year')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(avg_keywords):
        ax.text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    # Panel 3: Top keywords overall
    ax = axs[2]
    all_keywords = Counter()
    
    for year in years:
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                for talk in conference_data[year][talk_type]:
                    keywords = extract_keywords_from_talk(talk)
                    all_keywords.update(keywords)
    
    # Get top keywords
    top_kw = all_keywords.most_common(10)
    top_kw.reverse()  # For horizontal bar chart
    
    # Plot
    y_pos = range(len(top_kw))
    ax.barh(y_pos, [count for _, count in top_kw], color='salmon')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([kw for kw, _ in top_kw])
    ax.set_title('Top 10 Keywords Overall')
    ax.set_xlabel('Frequency')
    
    # Panel 4: Keyword diversity by year
    ax = axs[3]
    unique_keywords = []
    
    for year in years:
        # Get all keywords for this year
        year_keywords = Counter()
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in conference_data[year]:
                for talk in conference_data[year][talk_type]:
                    keywords = extract_keywords_from_talk(talk)
                    year_keywords.update(keywords)
        
        unique_keywords.append(len(year_keywords))
    
    # Plot
    ax.plot(years, unique_keywords, marker='o', linestyle='-', 
            linewidth=2, color='purple', markersize=8)
    ax.set_title('Number of Unique Keywords by Year')
    ax.set_ylabel('Count')
    ax.set_xlabel('Conference Year')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/keyword_QA_plots_basic.pdf')
    plt.close()

def create_keyword_wordcloud(conference_data):
    """Create a word cloud from the titles of all talks"""
    print("Creating keyword word cloud...")
    
    # Combine all titles
    all_titles = []
    
    for year, data in conference_data.items():
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                all_titles.extend([talk.get('Title', '') for talk in data[talk_type]])
    
    # Join all titles
    text = ' '.join(all_titles)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=2000, 
        height=1000,
        background_color='white',
        colormap='viridis',
        max_words=200,
        min_font_size=10,
        max_font_size=100
    ).generate(text)
    
    # Create figure
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/QM_keyword_wordcloud_basic.pdf')
    plt.close()

def analyze_conference_data(conference_data=None, output_dir='figures'):
    """Main function to analyze conference data and generate visualizations"""
    print("Analyzing conference data...")
    
    # First load the processed data if not provided
    if conference_data is None:
        try:
            print("\nLoading processed data...")
            conference_data = load_processed_data()
            if not conference_data:
                print("Error: Could not load processed data")
                return
        except Exception as e:
            print(f"Error loading processed data: {e}")
            traceback.print_exc()
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix unknown countries for known institutes
    try:
        print("\nFixing unknown countries for known institutes...")
        conference_data = fix_unknown_countries_for_known_institutes(conference_data)
    except Exception as e:
        print(f"Error fixing unknown countries: {e}")
        traceback.print_exc()
    
    # Pre-process the data
    print("\nPre-processing data...")
    try:
        filtered_data = preprocess_conference_data(conference_data)
    except Exception as e:
        print(f"Error pre-processing data: {e}")
        traceback.print_exc()
        filtered_data = conference_data

    print("\n===== BEGINNING ANALYSIS =====")
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data/analysis', exist_ok=True)
    
    # Display initial conference summary
    print("\nConference summary from processed data:")
    display_conference_summary(conference_data)
    
    # STEP 1: Fix inconsistencies in institute and country data
    print("\nSTEP 1: Fixing inconsistencies in institute and country data...")
    conference_data = fix_unknown_institute_country_data(conference_data)
    
    # STEP 2: Fix common affiliation problems
    print("\nSTEP 2: Fixing common affiliation problems...")
    conference_data = fix_common_affiliation_problems(conference_data)
    
    # STEP 3: Add manual country fixes
    print("\nSTEP 3: Adding manual country fixes...")
    conference_data = add_manual_country_fixes(conference_data)
    
    # STEP 4: Specifically fix unknown institutes
    print("\nSTEP 4: Specifically fixing unknown institutes...")
    conference_data = fix_unknown_institutes(conference_data)
    
    # Display final conference summary
    print("\nFinal conference summary:")
    display_conference_summary(conference_data)
    
    # STEP 5: Filter to only include relevant talk types
    print("\nSTEP 5: Filtering to include only relevant talk types...")
    filtered_data = filter_relevant_talk_types(conference_data)
    if not filtered_data:
        print("Error: Filtering failed, using original data")
        filtered_data = conference_data
    
    # STEP 6: Analyze keywords
    print("\nSTEP 6: Analyzing keywords...")
    
    # Keyword analysis
    try:
        print("Creating keyword analysis visualization...")
        analyze_keywords(filtered_data)
        print("Keyword analysis visualization created successfully!")
    except Exception as e:
        print(f"Error creating keyword analysis visualization: {e}")
        traceback.print_exc()
    
    # Keyword QA plots
    try:
        print("Creating keyword QA plots...")
        create_keyword_QA_plots(filtered_data)
        print("Keyword QA plots created successfully!")
    except Exception as e:
        print(f"Error creating keyword QA plots: {e}")
        traceback.print_exc()
    
    # Keyword word cloud
    try:
        print("Creating keyword word cloud...")
        create_keyword_wordcloud(filtered_data)
        print("Keyword word cloud created successfully!")
    except Exception as e:
        print(f"Error creating keyword word cloud: {e}")
        traceback.print_exc()
    
    print("\n===== ANALYSIS COMPLETE =====")
    print("All keyword visualizations have been saved to the 'figures' directory")
    
    return conference_data

# Add proper entry point at the end of the file
if __name__ == "__main__":
    analyze_conference_data() 