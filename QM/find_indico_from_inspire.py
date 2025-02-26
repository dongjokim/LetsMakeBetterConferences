import requests
import re

def get_qm_conferences():
    # List from https://sites.google.com/cougarnet.uh.edu/qm2023/previous-quark-matter-conferences
    conferences = [
        ("2023", "Houston, USA", "1139644"),  # QM2023
        ("2022", "Cracow, Poland", "895086"),  # QM2022
        ("2019", "Wuhan, China", "792436"),    # QM2019
        ("2018", "Venice, Italy", "656690"),   # QM2018
        ("2017", "Chicago (IL), USA", "433345"), # QM2017
        ("2015", "Kobe, Japan", "297045"),     # QM2015
        ("2014", "Darmstadt, Germany", "219436"), # QM2014
        ("2012", "Washington, USA", "181055"),  # QM2012
        ("2011", "Annecy, France", "137579"),  # QM2011
        ("2009", "Knoxville (TN), USA", None),
        ("2008", "Jaipur, India", None),
        ("2006", "Shanghai, China", None),
        ("2005", "Budapest, Hungary", None),
        ("2004", "Oakland (CA), USA", None),
        ("2002", "Nantes, France", None),
        ("2001", "Stony Brook (NY), USA", None),
        ("1999", "Torino, Italy", None),
        ("1997", "Tsukuba, Japan", None),
        ("1996", "Heidelberg, Germany", None),
        ("1995", "Monterey (CA), USA", None),
        ("1993", "Borlänge, Sweden", None),
        ("1991", "Gatlinburg (TN), USA", None),
        ("1990", "Menton, France", None),
        ("1988", "Lenox (MA), USA", None),
        ("1987", "Nordkirchen, Germany", None),
        ("1986", "Pacific Grove (CA), USA", None),
        ("1984", "Helsinki, Finland", None),
        ("1983", "Upton (NY), USA", None),
        ("1982", "Bielefeld, Germany", None),
    ]
    return conferences

def save_indico_ids():
    conferences = get_qm_conferences()
    
    # Save to file
    with open('listofQMindigo', 'w') as f:
        for year, location, indico_id in conferences:
            if indico_id:  # Only save conferences with known Indico IDs
                f.write(f"{year} {indico_id}\n")
                print(f"Saved QM{year} ({location}) - Indico ID: {indico_id}")
    
    print("\nSaved IDs to listofQMindigo")

if __name__ == "__main__":
    save_indico_ids() 