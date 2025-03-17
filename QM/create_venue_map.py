import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Conference venues data
venues = {
    2011: {'city': 'Annecy', 'country': 'France', 'continent': 'Europe', 'lat': 45.899, 'lon': 6.129},
    2012: {'city': 'Washington DC', 'country': 'USA', 'continent': 'North America', 'lat': 38.907, 'lon': -77.037},
    2014: {'city': 'Darmstadt', 'country': 'Germany', 'continent': 'Europe', 'lat': 49.872, 'lon': 8.651},
    2015: {'city': 'Kobe', 'country': 'Japan', 'continent': 'Asia', 'lat': 34.690, 'lon': 135.196},
    2017: {'city': 'Chicago', 'country': 'USA', 'continent': 'North America', 'lat': 41.878, 'lon': -87.630},
    2018: {'city': 'Venice', 'country': 'Italy', 'continent': 'Europe', 'lat': 45.438, 'lon': 12.326},
    2019: {'city': 'Wuhan', 'country': 'China', 'continent': 'Asia', 'lat': 30.593, 'lon': 114.306},
    2022: {'city': 'Krakow', 'country': 'Poland', 'continent': 'Europe', 'lat': 50.064, 'lon': 19.944},
    2023: {'city': 'Houston', 'country': 'USA', 'continent': 'North America', 'lat': 29.760, 'lon': -95.370},
    2025: {'city': 'Frankfurt', 'country': 'Germany', 'continent': 'Europe', 'lat': 50.110, 'lon': 8.682}
}

# Create a DataFrame
venues_df = pd.DataFrame.from_dict(venues, orient='index')

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Create figure with world map
fig = plt.figure(figsize=(15, 10), dpi=300)

# Main world map
ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax_map.set_global()

# Add map features
ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

# Define colors for continents
continent_colors = {
    'Europe': '#4285F4',       # Blue
    'North America': '#EA4335', # Red
    'Asia': '#FBBC05',         # Yellow
}

# Plot each venue with a numbered marker
for i, (year, venue) in enumerate(sorted(venues.items())):
    color = continent_colors[venue['continent']]
    
    # Adjust positions for Darmstadt and Frankfurt to avoid overlap
    plot_lon, plot_lat = venue['lon'], venue['lat']
    
    if venue['city'] == 'Darmstadt':
        # Shift Darmstadt to the southwest, but less than before
        plot_lon -= 1.5
        plot_lat -= 1.0
    elif venue['city'] == 'Frankfurt':
        # Shift Frankfurt to the northeast, but less than before
        plot_lon += 1.5
        plot_lat += 1.0
    elif venue['city'] == 'Annecy':
        # Shift Annecy more significantly to the west
        plot_lon -= 3.0
        plot_lat -= 0.5
    
    # Set marker size based on continent (smaller for Europe)
    marker_size = 120 if venue['continent'] != 'Europe' else 100
    font_size = 9 if venue['continent'] != 'Europe' else 8
    
    # Plot marker with number
    ax_map.scatter(plot_lon, plot_lat, s=marker_size, color=color, 
                  edgecolor='white', linewidth=1, zorder=5, alpha=0.8,
                  transform=ccrs.PlateCarree())
    
    # Add number in the center of the marker
    ax_map.text(plot_lon, plot_lat, str(i+1), 
               fontsize=font_size, ha='center', va='center', color='white', 
               fontweight='bold', transform=ccrs.PlateCarree(), zorder=6)
    
    # If we shifted the marker, draw a thin line to the actual location
    if venue['city'] in ['Darmstadt', 'Frankfurt', 'Annecy']:
        # Draw a more visible connecting line
        ax_map.plot([plot_lon, venue['lon']], [plot_lat, venue['lat']], 
                   color='gray', linewidth=0.8, alpha=0.7, zorder=4,
                   transform=ccrs.PlateCarree())
        
        # Add a small dot at the actual location
        ax_map.scatter(venue['lon'], venue['lat'], s=20, color='gray', 
                      edgecolor='white', linewidth=0.5, zorder=3, alpha=0.6,
                      transform=ccrs.PlateCarree())

# Connect venues with great circle paths
years = sorted(venues.keys())
for i in range(len(years)-1):
    year1, year2 = years[i], years[i+1]
    venue1, venue2 = venues[year1], venues[year2]
    
    # Draw a great circle path
    ax_map.plot([venue1['lon'], venue2['lon']], [venue1['lat'], venue2['lat']], 
               color='gray', linewidth=1.5, alpha=0.6, zorder=3,
               transform=ccrs.Geodetic())

# Create custom legend handles for venues
venue_handles = []
for i, (year, venue) in enumerate(sorted(venues.items())):
    color = continent_colors[venue['continent']]
    
    # Create a simple marker for the legend
    marker = Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=color, markeredgecolor='white',
                   markersize=15, label=f"{i+1}. {venue['city']} ({year})")
    
    venue_handles.append(marker)

# Create legend for continents
continent_handles = []
for continent, color in continent_colors.items():
    continent_handles.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markeredgecolor='white',
                                  markersize=10, label=continent))

# Add both legends to the map
# First the venue legend (positioned at the left)
venue_legend = ax_map.legend(handles=venue_handles, 
                           loc='center left', 
                           frameon=True,
                           framealpha=0.9,
                           fontsize=9,
                           title='Conference Venues',
                           title_fontsize=12)

# Then the continent legend (positioned below the venue legend)
ax_map.add_artist(venue_legend)  # Keep the first legend
continent_legend = ax_map.legend(handles=continent_handles, 
                               loc='lower left', 
                               frameon=True,
                               framealpha=0.9,
                               fontsize=10,
                               title='Continents',
                               title_fontsize=10)

# Add title to map
ax_map.set_title('Geographical Distribution of Quark Matter Conference Venues (2011-2025)', 
         fontsize=16, fontweight='bold')

# Save the figure
plt.savefig('figures/conference_venues.pdf', bbox_inches='tight')
plt.savefig('figures/conference_venues.png', bbox_inches='tight', dpi=300)
plt.close()

print("Conference venue map created successfully!") 