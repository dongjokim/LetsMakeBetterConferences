# QM Conference Data Analysis Tool

This README provides instructions for using the Quark Matter (QM) Conference Data Analysis Tool, which extracts and processes information from past QM conferences using the CERN Indico API.

## Overview

This tool extracts contribution data (talks, posters, etc.) from Quark Matter conferences, processes speaker affiliations and countries, and generates structured datasets for further analysis. The data includes speaker information, talk titles, abstracts, session types, and more.

## Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - requests
  - pandas
  - json
  - re
  - csv
  - os
  - sys
  - time
  - matplotlib (for analysis visualization)
  - numpy (for data analysis)

## Installation

1. Clone this repository or download the source code
2. Install required packages:

```bash
pip install requests pandas matplotlib numpy
```

3. Ensure you have the following files in your directory structure:
   - `generate_conference_data.py` - Main script for data extraction
   - `fetch_participants.py` - Script to fetch participant data
   - `analyze_conference_data.py` - Script to analyze and visualize data
   - `listofQMindigo` - File with conference Indico IDs
   - `institute_country_database.csv` (optional) - Institution to country mappings

## Directory Structure

Create the following directories if they don't exist:

```
data/
├── processed/
├── participants/
└── figures/     # For visualization outputs
```

## Usage

### Execution Order

Run the scripts in the following order:

```bash
# 1. First, fetch participant data from Indico:
python fetch_participants.py

# 2. Then, fetch and process all conference data:
python generate_conference_data.py

# 3. Finally, analyze the data and generate visualizations:
python analyze_conference_data.py
```

### Step 1: Fetch Participants

This step collects participant data from each conference using the Indico API:

```bash
python fetch_participants.py
```

Expected output:
- Creates `data/participants/all_participants.csv`
- Console will show progress for each conference year

### Step 2: Generate Conference Data

This step fetches detailed contribution data and processes it:

```bash
python generate_conference_data.py
```

Expected output:
- Creates processed data files in `data/processed/` directory
- Generates statistics and unknown institute lists
- Console will show processing progress for each conference

### Step 3: Analyze Conference Data

This step performs analysis on the processed data and creates visualizations:

```bash
python analyze_conference_data.py
```

Expected output:
- Creates visualization files in `data/figures/` directory
- Generates additional statistics
- Console will show analysis progress

### Understanding the Input Files

#### `listofQMindigo`

This file contains year and Indico ID mappings for QM conferences in the format:
```
# Year IndicoID
2011 181055
2012 181214
...
```

Lines starting with `#` are comments.

#### `institute_country_database.csv` (optional)

If provided, this file maps institute names to countries in CSV format:
```
Institute,Country
University of Jyvaskyla,Finland
CERN,Switzerland
...
```

This helps the tool correctly associate speakers with their countries.

### Output Files and Data Structure

The tool generates several outputs:

1. **Processed Conference Data**:
   - `data/processed_conference_data.json` - Combined data for all conferences
   - `data/processed/{year}/` - Individual conference data by year
     - `all_talks.csv` - All contributions with fields:
       - Session, Type, Title, Speaker, Institute, Country, Abstract
     - `plenary_talks.csv` - Plenary sessions
     - `parallel_talks.csv` - Parallel sessions
     - `poster_talks.csv` - Poster sessions
     - `statistics.json` - Summary statistics including counts by type, country

2. **Unknown Institutes**:
   - `unknown_institutes.txt` - List of institutes needing country mapping

3. **Statistics Reports**:
   - `final_statistics_report.txt` - Detailed statistics on processed data

4. **Participant Data**:
   - `data/participants/all_participants.csv` - Contains participant information

5. **Analysis Results**:
   - `data/figures/` - Contains visualizations and analysis results
     - Country distributions
     - Session type distributions
     - Year-over-year trends
     - Topic analysis

## Troubleshooting

### API Connection Issues

If you encounter API connection errors:
- Check your internet connection
- Verify the Indico IDs in `listofQMindigo` are correct
- Consider using a VPN if CERN's API is blocked at your location
- Add delay between requests using the `--delay` parameter

### Missing Affiliations

If many talks have unknown affiliations:
1. Check the `unknown_institutes.txt` file
2. Add mappings to `institute_country_database.csv`
3. Run the script again

### Cached Data Issues

To force reprocessing of data even if cached data exists:
- Delete the `data/processed_conference_data.json` file
- Use the `--force-refresh` flag when running the script

### Common Error Messages

- "API rate limit exceeded": Wait a few minutes and try again
- "Unknown institute": Add the institute to `institute_country_database.csv`
- "Conference ID not found": Check the `listofQMindigo` file for accuracy

## Advanced Usage

### Command Line Arguments

Both scripts support several command line arguments:

```bash
# Process only specific years
python generate_conference_data.py --years 2018 2019 2022

# Force refresh of cached data
python generate_conference_data.py --force-refresh

# Add delay between API requests (in seconds)
python fetch_participants.py --delay 2
```

### Extending the Code

Key functions you might want to modify:
- `categorize_session()` - Logic for categorizing session types
- `extract_country()` - Logic for extracting countries from affiliations
- `should_reprocess_data()` - Controls when data is reprocessed
- `analyze_topics()` - Modify topic analysis algorithms

## License

This software is provided for academic research purposes only. Use at your own risk.

## Acknowledgments

This tool uses the CERN Indico API for retrieving conference data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
