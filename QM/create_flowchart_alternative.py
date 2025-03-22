import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_flow_chart():
    """
    Creates a flow chart visualization of the fetch_and_analyze_conferences.py code structure
    using matplotlib, which doesn't require the Graphviz software.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Define colors
    light_blue = '#e6f3ff'
    light_green = '#c5e8d5'
    light_yellow = '#fffacd'
    light_red = '#ffcccb'
    
    # Define box height and width
    box_width = 0.3
    box_height = 0.05
    cluster_padding = 0.01
    phase_height = 0.15  # Height of each phase section
    
    # Define starting positions
    start_x = 0.35
    current_y = 0.95  # Start from top
    
    # Draw title
    ax.text(0.5, 0.98, 'Flow Chart of fetch_and_analyze_conferences.py', 
            horizontalalignment='center', fontsize=16, fontweight='bold')
    
    # Draw start node
    start_oval = patches.Ellipse((start_x, current_y), box_width, box_height, 
                              fill=True, facecolor=light_green, edgecolor='black')
    ax.add_patch(start_oval)
    ax.text(start_x, current_y, 'Start', horizontalalignment='center', 
            verticalalignment='center', fontsize=12)
    
    current_y -= 0.06
    
    # Draw PHASE 1: DATA COLLECTION
    phase1_y = current_y
    phase1_rect = patches.Rectangle((0.1, current_y - phase_height), 0.8, phase_height, 
                              fill=True, facecolor=light_blue, edgecolor='gray', alpha=0.5)
    ax.add_patch(phase1_rect)
    ax.text(0.5, current_y - 0.02, 'PHASE 1: DATA COLLECTION', 
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    current_y -= 0.05
    
    # Draw Load Indico IDs box
    indico_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                   box_width, box_height, fill=True, 
                                   facecolor=light_blue, edgecolor='black')
    ax.add_patch(indico_rect)
    ax.text(start_x, current_y, 'Load Indico IDs', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to next box
    current_y -= 0.06
    ax.arrow(start_x, current_y + 0.03, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw check IDs diamond using a rotated rectangle instead of Diamond
    diamond_width = box_width * 1.2
    diamond_height = box_height * 1.2
    
    # Create a rotated rectangle to make a diamond shape
    diamond_rect = patches.Rectangle(
        (start_x - diamond_width/4, current_y - diamond_height/4),
        diamond_width/2, diamond_height/2, 
        angle=45, 
        fill=True, 
        facecolor=light_yellow, 
        edgecolor='black'
    )
    ax.add_patch(diamond_rect)
    ax.text(start_x, current_y, 'IDs loaded?', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to error exit
    exit_x = start_x + 0.25
    ax.arrow(start_x + 0.06, current_y, 0.10, 0, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    ax.text(start_x + 0.1, current_y + 0.02, 'No', fontsize=8)
    
    # Draw error exit oval
    exit_oval = patches.Ellipse((exit_x, current_y), box_width, box_height, 
                               fill=True, facecolor=light_red, edgecolor='black')
    ax.add_patch(exit_oval)
    ax.text(exit_x, current_y, 'Exit with error', horizontalalignment='center', 
            verticalalignment='center', fontsize=10)
    
    # Continue with "Yes" path
    current_y -= 0.10
    ax.arrow(start_x, current_y + 0.08, 0, -0.03, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    ax.text(start_x - 0.04, current_y + 0.05, 'Yes', fontsize=8)
    
    # Draw PHASE 2: DATA LOADING
    phase2_y = current_y
    phase2_rect = patches.Rectangle((0.1, current_y - phase_height), 0.8, phase_height, 
                              fill=True, facecolor=light_blue, edgecolor='gray', alpha=0.5)
    ax.add_patch(phase2_rect)
    ax.text(0.5, current_y - 0.02, 'PHASE 2: DATA LOADING', 
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    # Draw Load Registration Data box
    current_y -= 0.05
    load_reg_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                    box_width, box_height, fill=True, 
                                    facecolor=light_blue, edgecolor='black')
    ax.add_patch(load_reg_rect)
    ax.text(start_x, current_y, 'Load Registration Data', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to next box
    current_y -= 0.06
    ax.arrow(start_x, current_y + 0.03, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Load Plenary Talks box
    load_plenary_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                        box_width, box_height, fill=True, 
                                        facecolor=light_blue, edgecolor='black')
    ax.add_patch(load_plenary_rect)
    ax.text(start_x, current_y, 'Load Plenary Talks', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to next box
    current_y -= 0.06
    ax.arrow(start_x, current_y + 0.03, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Load Parallel Talks box
    load_parallel_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                         box_width, box_height, fill=True, 
                                         facecolor=light_blue, edgecolor='black')
    ax.add_patch(load_parallel_rect)
    ax.text(start_x, current_y, 'Load Parallel Talks', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to next box
    current_y -= 0.06
    ax.arrow(start_x, current_y + 0.03, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Load Poster Talks box
    load_poster_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                       box_width, box_height, fill=True, 
                                       facecolor=light_blue, edgecolor='black')
    ax.add_patch(load_poster_rect)
    ax.text(start_x, current_y, 'Load Poster Talks', horizontalalignment='center',
           verticalalignment='center', fontsize=10)
    
    # Draw arrow to next phase
    current_y -= 0.10
    ax.arrow(start_x, current_y + 0.08, 0, -0.03, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw PHASE 3: DATA ENHANCEMENT
    phase3_y = current_y
    phase3_rect = patches.Rectangle((0.1, current_y - phase_height), 0.8, phase_height, 
                              fill=True, facecolor=light_blue, edgecolor='gray', alpha=0.5)
    ax.add_patch(phase3_rect)
    ax.text(0.5, current_y - 0.02, 'PHASE 3: DATA ENHANCEMENT', 
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    # Draw Enhance Institute Data box
    current_y -= 0.05
    enhance_inst_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                        box_width, box_height*1.2, fill=True, 
                                        facecolor=light_blue, edgecolor='black')
    ax.add_patch(enhance_inst_rect)
    ax.text(start_x, current_y, 'Enhance Institute Data\n(enhance_institute_data())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw function detail box
    detail_x = start_x + 0.25
    detail_rect = patches.Rectangle((detail_x - 0.15, current_y - 0.05), 
                                   0.3, 0.1, fill=True, 
                                   facecolor=light_yellow, edgecolor='black', alpha=0.7)
    ax.add_patch(detail_rect)
    ax.text(detail_x, current_y, 'enhance_institute_data():\n- Maps speakers to institutes\n- Updates missing affiliations\n- Provides nationality info', 
            horizontalalignment='center', verticalalignment='center', fontsize=7)
    
    # Draw dashed line connecting to detail
    ax.plot([start_x + box_width/2, detail_x - 0.15], [current_y, current_y], 
            'k--', linewidth=0.5)
    
    # Draw arrow to next box
    current_y -= 0.08
    ax.arrow(start_x, current_y + 0.05, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Fill Missing Affiliations box
    fill_missing_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                        box_width, box_height*1.2, fill=True, 
                                        facecolor=light_blue, edgecolor='black')
    ax.add_patch(fill_missing_rect)
    ax.text(start_x, current_y, 'Fill Missing Affiliations\n(fill_missing_affiliations())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw arrow to next phase
    current_y -= 0.10
    ax.arrow(start_x, current_y + 0.08, 0, -0.03, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw PHASE 4: VISUALIZATION
    phase4_y = current_y
    phase4_rect = patches.Rectangle((0.1, current_y - phase_height), 0.8, phase_height, 
                              fill=True, facecolor=light_blue, edgecolor='gray', alpha=0.5)
    ax.add_patch(phase4_rect)
    ax.text(0.5, current_y - 0.02, 'PHASE 4: VISUALIZATION', 
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    # Draw Create Institute Plots box
    current_y -= 0.05
    inst_plots_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                      box_width, box_height*1.2, fill=True, 
                                      facecolor=light_blue, edgecolor='black')
    ax.add_patch(inst_plots_rect)
    ax.text(start_x, current_y, 'Create Institute Plots\n(create_institute_plots())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw function detail box
    detail_x = start_x + 0.25
    detail_rect = patches.Rectangle((detail_x - 0.15, current_y - 0.05), 
                                   0.3, 0.1, fill=True, 
                                   facecolor=light_yellow, edgecolor='black', alpha=0.7)
    ax.add_patch(detail_rect)
    ax.text(detail_x, current_y, 'create_institute_plots():\n- Plots for plenary talks\n- Plots for parallel talks\n- Combined bubble charts', 
            horizontalalignment='center', verticalalignment='center', fontsize=7)
    
    # Draw dashed line connecting to detail
    ax.plot([start_x + box_width/2, detail_x - 0.15], [current_y, current_y], 
            'k--', linewidth=0.5)
    
    # Draw arrow to next box
    current_y -= 0.08
    ax.arrow(start_x, current_y + 0.05, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Create Country Plots box
    country_plots_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                         box_width, box_height*1.2, fill=True, 
                                         facecolor=light_blue, edgecolor='black')
    ax.add_patch(country_plots_rect)
    ax.text(start_x, current_y, 'Create Country Plots\n(create_country_plots())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw arrow to next box
    current_y -= 0.08
    ax.arrow(start_x, current_y + 0.05, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Create Gender Plots box
    gender_plots_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                        box_width, box_height*1.2, fill=True, 
                                        facecolor=light_blue, edgecolor='black')
    ax.add_patch(gender_plots_rect)
    ax.text(start_x, current_y, 'Create Gender Plots\n(create_gender_plots())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw arrow to next phase
    current_y -= 0.10
    ax.arrow(start_x, current_y + 0.08, 0, -0.03, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw PHASE 5: OUTPUT
    phase5_y = current_y
    phase5_rect = patches.Rectangle((0.1, current_y - phase_height), 0.8, phase_height, 
                              fill=True, facecolor=light_blue, edgecolor='gray', alpha=0.5)
    ax.add_patch(phase5_rect)
    ax.text(0.5, current_y - 0.02, 'PHASE 5: OUTPUT', 
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    # Draw Export Processed Data box
    current_y -= 0.05
    export_data_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                       box_width, box_height*1.2, fill=True, 
                                       facecolor=light_blue, edgecolor='black')
    ax.add_patch(export_data_rect)
    ax.text(start_x, current_y, 'Export Processed Data\n(export_processed_data())', 
            horizontalalignment='center', verticalalignment='center', fontsize=9)
    
    # Draw arrow to next box
    current_y -= 0.08
    ax.arrow(start_x, current_y + 0.05, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw Generate Summary Stats box
    summary_stats_rect = patches.Rectangle((start_x - box_width/2, current_y - box_height/2), 
                                         box_width, box_height*1.2, fill=True, 
                                         facecolor=light_blue, edgecolor='black')
    ax.add_patch(summary_stats_rect)
    ax.text(start_x, current_y, 'Generate Summary Statistics\n(generate_summary_statistics())', 
            horizontalalignment='center', verticalalignment='center', fontsize=8)
    
    # Draw arrow to End
    current_y -= 0.08
    ax.arrow(start_x, current_y + 0.05, 0, -0.02, head_width=0.01, head_length=0.01, 
            fc='black', ec='black')
    
    # Draw End node
    end_oval = patches.Ellipse((start_x, current_y), box_width, box_height, 
                             fill=True, facecolor=light_green, edgecolor='black')
    ax.add_patch(end_oval)
    ax.text(start_x, current_y, 'End', horizontalalignment='center', 
            verticalalignment='center', fontsize=12)
    
    # Remove axes ticks and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('flow_chart_fetch_and_analyze_conferences.png', dpi=300, bbox_inches='tight')
    print("Flow chart created: flow_chart_fetch_and_analyze_conferences.png")
    plt.close()

if __name__ == "__main__":
    create_flow_chart() 