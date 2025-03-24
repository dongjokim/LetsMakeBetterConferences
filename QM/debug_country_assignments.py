#!/usr/bin/env python3
"""
debug_country_assignments.py - Debug script to analyze country assignments in QM conference data
Specifically designed to investigate potentially inflated USA contribution counts.
"""

import json
import os
import pandas as pd
import re
from collections import Counter

def load_processed_data():
    """Load the processed conference data from JSON file"""
    try:
        with open('data/processed_conference_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def analyze_usa_assignments(conference_data):
    """Analyze USA assignments in the conference data"""
    # Collect all USA-assigned contributions for detailed analysis
    usa_contributions = []
    other_contributions = []
    
    # Also keep track of yearly counts for trends
    usa_by_year = {}
    total_by_year = {}
    
    for year, data in conference_data.items():
        usa_count = 0
        total_count = 0
        
        for talk_type in ['plenary_talks', 'parallel_talks', 'poster_talks']:
            if talk_type in data:
                for talk in data[talk_type]:
                    total_count += 1
                    
                    # Check if assigned to USA
                    country = talk.get('Country', 'Unknown')
                    if country == 'USA':
                        usa_count += 1
                        # Store for detailed analysis
                        talk_copy = talk.copy()
                        talk_copy['Year'] = year
                        talk_copy['Talk_Type'] = talk_type
                        usa_contributions.append(talk_copy)
                    else:
                        talk_copy = talk.copy()
                        talk_copy['Year'] = year
                        talk_copy['Talk_Type'] = talk_type
                        other_contributions.append(talk_copy)
        
        usa_by_year[year] = usa_count
        total_by_year[year] = total_count
    
    return {
        'usa_contributions': usa_contributions,
        'other_contributions': other_contributions,
        'usa_by_year': usa_by_year,
        'total_by_year': total_by_year
    }

def analyze_affiliation_patterns(contributions):
    """Analyze patterns in affiliations to identify potential parsing issues"""
    affiliation_patterns = Counter()
    
    for talk in contributions:
        affiliation = talk.get('Institute', '')
        if not affiliation:
            affiliation = talk.get('Affiliation', '')
            
        if affiliation:
            # Look for common words/patterns that might be causing issues
            affiliation_lower = affiliation.lower()
            
            # Check for specific patterns
            if 'collaboration' in affiliation_lower or 'collab' in affiliation_lower:
                affiliation_patterns['Contains Collaboration'] += 1
                
            if re.search(r'usa|united states|america', affiliation_lower):
                affiliation_patterns['Contains USA keyword'] += 1
                
            if ',' in affiliation:
                affiliation_patterns['Contains comma'] += 1
                
            # Check for university but no country
            if 'university' in affiliation_lower and not re.search(r'usa|united states|america|japan|germany|france|uk|italy|china', affiliation_lower):
                affiliation_patterns['University without country'] += 1
    
    return affiliation_patterns

def identify_outlier_years(usa_by_year, total_by_year):
    """Identify years with unusually high USA percentages"""
    outliers = []
    
    # Calculate percentage for each year
    percentages = {}
    for year in usa_by_year:
        total = total_by_year.get(year, 0)
        if total > 0:
            percentage = (usa_by_year[year] / total) * 100
            percentages[year] = percentage
    
    # Calculate average percentage
    avg_percentage = sum(percentages.values()) / len(percentages) if percentages else 0
    
    # Identify outliers (>20% deviation from average)
    for year, percentage in percentages.items():
        if percentage > avg_percentage * 1.2:  # 20% higher than average
            outliers.append((year, percentage, avg_percentage))
    
    return outliers, percentages

def sample_suspicious_affiliations(usa_contributions, n=20):
    """Sample potentially suspicious USA assignments for manual review"""
    suspicious = []
    
    for talk in usa_contributions:
        affiliation = talk.get('Institute', talk.get('Affiliation', ''))
        if affiliation:
            # Check for suspicious patterns
            if not re.search(r'usa|united states|america', affiliation.lower()):
                suspicious.append((talk['Year'], talk['Talk_Type'], talk.get('Speaker', ''), affiliation))
            elif re.search(r'collaboration|collab', affiliation.lower()):
                suspicious.append((talk['Year'], talk['Talk_Type'], talk.get('Speaker', ''), affiliation))
    
    # Return a sample
    return suspicious[:n]

def check_institute_database():
    """Check the institute_country_database.csv for USA assignments"""
    usa_institutes = []
    
    try:
        if os.path.exists('institute_country_database.csv'):
            df = pd.read_csv('institute_country_database.csv')
            
            # Find all institutes assigned to USA
            for _, row in df.iterrows():
                if 'USA' in str(row.get('Country', '')):
                    usa_institutes.append(row.get('Institute', ''))
    except Exception as e:
        print(f"Error checking institute database: {e}")
    
    return usa_institutes

def main():
    """Main function to run the debug analysis"""
    print("Starting country assignment debug analysis...")
    
    # Load the processed data
    conference_data = load_processed_data()
    if not conference_data:
        print("Failed to load conference data. Exiting.")
        return
    
    # Analyze USA assignments
    results = analyze_usa_assignments(conference_data)
    
    # Print overall stats
    total_talks = sum(results['total_by_year'].values())
    total_usa = sum(results['usa_by_year'].values())
    print(f"\nOverall USA Statistics:")
    print(f"Total Talks: {total_talks}")
    print(f"USA Talks: {total_usa} ({total_usa/total_talks*100:.2f}%)")
    
    # Identify outlier years
    outliers, percentages = identify_outlier_years(results['usa_by_year'], results['total_by_year'])
    
    print("\nUSA Percentage by Year:")
    for year in sorted(percentages.keys()):
        print(f"  {year}: {percentages[year]:.2f}% ({results['usa_by_year'][year]} of {results['total_by_year'][year]})")
    
    if outliers:
        print("\nOutlier Years (>20% deviation from average):")
        for year, percentage, avg in outliers:
            print(f"  {year}: {percentage:.2f}% (average: {avg:.2f}%)")
    
    # Analyze affiliation patterns
    print("\nAnalyzing affiliation patterns in USA contributions...")
    usa_patterns = analyze_affiliation_patterns(results['usa_contributions'])
    
    print("\nCommon Patterns in USA Affiliations:")
    for pattern, count in usa_patterns.most_common():
        print(f"  {pattern}: {count} ({count/len(results['usa_contributions'])*100:.2f}%)")
    
    # Sample suspicious affiliations
    suspicious = sample_suspicious_affiliations(results['usa_contributions'])
    
    print("\nSample of Suspicious USA Assignments:")
    for year, talk_type, speaker, affiliation in suspicious:
        print(f"  {year} | {talk_type} | {speaker} | {affiliation}")
    
    # Check institute database
    usa_institutes = check_institute_database()
    
    print(f"\nUSA Institutes in Database: {len(usa_institutes)}")
    print("Sample of institutes mapped to USA:")
    for institute in usa_institutes[:10]:  # Show first 10
        print(f"  {institute}")
    
    # Generate recommendations
    print("\nRecommendations:")
    if suspicious:
        print("1. Review suspicious USA assignments manually")
    
    if usa_patterns.get('Contains Collaboration', 0) > 0:
        print("2. Improve handling of collaborations in country assignment logic")
    
    if usa_patterns.get('University without country', 0) > 0:
        print("3. Add more university-to-country mappings in institute_country_database.csv")
    
    print("4. Review extract_country() function for potential USA bias")
    print("5. Consider enhancing the institute normalization to handle more variants")

if __name__ == "__main__":
    main()