import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime

def load_conference_data():
    conferences = []
    data_dir = 'data'
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('_data.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                conferences.append(json.load(f))
    
    return conferences

def analyze_trends():
    # Load all conference data
    conferences = load_conference_data()
    
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Prepare data structures for analysis
    stats = []
    
    for conf in conferences:
        year = conf['metadata']['year']
        print(f"Analyzing QM{year}...")
        
        contributions = conf['results'][0].get('contributions', [])
        
        # Extract statistics
        stat = {
            'year': int(year),
            'total_talks': len(contributions),
            'speakers': set(),
            'institutions': set(),
            'countries': Counter(),
            'session_types': Counter()
        }
        
        for contrib in contributions:
            if 'speaker' in contrib:
                speaker = contrib['speaker'].get('name', '')
                institution = contrib['speaker'].get('affiliation', '')
                country = contrib['speaker'].get('country', '')
                session = contrib.get('session', '')
                
                if speaker:
                    stat['speakers'].add(speaker)
                if institution:
                    stat['institutions'].add(institution)
                if country:
                    stat['countries'][country] += 1
                if session:
                    stat['session_types'][session] += 1
        
        # Convert sets to counts
        stat['unique_speakers'] = len(stat['speakers'])
        stat['unique_institutions'] = len(stat['institutions'])
        
        stats.append(stat)
    
    # Sort by year
    stats.sort(key=lambda x: x['year'])
    
    # Create visualizations
    
    # 1. Total talks trend
    plt.figure(figsize=(12, 6))
    years = [s['year'] for s in stats]
    talks = [s['total_talks'] for s in stats]
    plt.plot(years, talks, 'o-')
    plt.title('Number of Talks per Conference')
    plt.xlabel('Year')
    plt.ylabel('Number of Talks')
    plt.grid(True)
    plt.savefig('analysis/talks_trend.pdf')
    plt.close()
    
    # 2. Diversity metrics
    plt.figure(figsize=(12, 6))
    speakers = [s['unique_speakers'] for s in stats]
    institutions = [s['unique_institutions'] for s in stats]
    plt.plot(years, speakers, 'o-', label='Unique Speakers')
    plt.plot(years, institutions, 'o-', label='Unique Institutions')
    plt.title('Conference Diversity Metrics')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/diversity_trend.pdf')
    plt.close()
    
    # 3. Geographic distribution evolution
    # Create a heatmap of top 15 countries over time
    all_countries = Counter()
    for s in stats:
        all_countries.update(s['countries'])
    
    top_countries = [c for c, _ in all_countries.most_common(15)]
    country_data = []
    
    for s in stats:
        year_data = [s['countries'].get(country, 0) for country in top_countries]
        country_data.append(year_data)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(np.array(country_data).T, aspect='auto')
    plt.colorbar(label='Number of Talks')
    plt.yticks(range(len(top_countries)), top_countries)
    plt.xticks(range(len(years)), years, rotation=45)
    plt.title('Geographic Distribution Evolution')
    plt.tight_layout()
    plt.savefig('analysis/geographic_evolution.pdf')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("==================")
    print(f"Total conferences analyzed: {len(stats)}")
    print(f"Average talks per conference: {np.mean([s['total_talks'] for s in stats]):.1f}")
    print(f"Average unique speakers: {np.mean([s['unique_speakers'] for s in stats]):.1f}")
    print(f"Average unique institutions: {np.mean([s['unique_institutions'] for s in stats]):.1f}")
    
    # Save processed data
    with open('analysis/processed_stats.json', 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'conferences': [{
                'year': s['year'],
                'total_talks': s['total_talks'],
                'unique_speakers': s['unique_speakers'],
                'unique_institutions': s['unique_institutions'],
                'top_countries': dict(s['countries'].most_common(10)),
                'session_types': dict(s['session_types'].most_common())
            } for s in stats]
        }, f, indent=2)

if __name__ == "__main__":
    analyze_trends() 