def create_theory_experiment_balance_plot(conference_data):
    """Create visualization showing the balance between theory and experiment presentations"""
    print("Creating theory-experiment balance visualization...")
    
    # Define keywords that indicate theoretical or experimental work
    theory_keywords = [
        'theory', 'theoretical', 'model', 'models', 'simulation', 'simulations', 
        'calculation', 'calculations', 'lattice', 'qcd', 'effective field theory',
        'eft', 'hydrodynamic', 'hydro', 'transport', 'monte carlo', 'perturbative',
        'non-perturbative', 'framework', 'approach', 'formalism', 'equation of state',
        'eos', 'viscosity', 'predict', 'prediction', 'predicted', 'microscopic'
    ]
    
    experiment_keywords = [
        'experiment', 'experimental', 'measurement', 'measurements', 'data',
        'results', 'observed', 'observation', 'detector', 'detectors', 'measured',
        'alice', 'atlas', 'cms', 'lhcb', 'star', 'phenix', 'brahms', 'phobos',
        'reconstruction', 'trigger', 'calibration', 'analysis', 'beam', 'collision',
        'luminosity', 'run', 'preliminary', 'calorimeter', 'spectrometer', 'tracking'
    ]
    
    # Extract relevant data for each year
    years = sorted([year for year in conference_data.keys() if year != '2025'])
    theory_counts = []
    experiment_counts = []
    both_counts = []
    other_counts = []
    total_counts = []
    
    # Also track by presentation type
    plenary_theory = []
    plenary_experiment = []
    plenary_both = []
    parallel_theory = []
    parallel_experiment = []
    parallel_both = []
    
    for year in years:
        if year in conference_data:
            # For title-based classification
            theory_count = 0
            experiment_count = 0
            both_count = 0
            other_count = 0
            total_count = 0
            
            # For presentation type tracking
            plenary_theory_count = 0
            plenary_experiment_count = 0
            plenary_both_count = 0
            parallel_theory_count = 0
            parallel_experiment_count = 0
            parallel_both_count = 0
            
            # Process plenary talks
            if 'plenary_talks' in conference_data[year]:
                for talk in conference_data[year]['plenary_talks']:
                    if not talk.get('Title'):
                        continue
                        
                    total_count += 1
                    title_lower = talk.get('Title', '').lower()
                    
                    # Check for theory/experiment keywords in title
                    is_theory = any(keyword in title_lower for keyword in theory_keywords)
                    is_experiment = any(keyword in title_lower for keyword in experiment_keywords)
                    
                    if is_theory and is_experiment:
                        both_count += 1
                        plenary_both_count += 1
                    elif is_theory:
                        theory_count += 1
                        plenary_theory_count += 1
                    elif is_experiment:
                        experiment_count += 1
                        plenary_experiment_count += 1
                    else:
                        other_count += 1
            
            # Process parallel talks
            if 'parallel_talks' in conference_data[year]:
                for talk in conference_data[year]['parallel_talks']:
                    if not talk.get('Title'):
                        continue
                        
                    total_count += 1
                    title_lower = talk.get('Title', '').lower()
                    
                    # Check for theory/experiment keywords in title
                    is_theory = any(keyword in title_lower for keyword in theory_keywords)
                    is_experiment = any(keyword in title_lower for keyword in experiment_keywords)
                    
                    if is_theory and is_experiment:
                        both_count += 1
                        parallel_both_count += 1
                    elif is_theory:
                        theory_count += 1
                        parallel_theory_count += 1
                    elif is_experiment:
                        experiment_count += 1
                        parallel_experiment_count += 1
                    else:
                        other_count += 1
            
            # Store counts
            theory_counts.append(theory_count)
            experiment_counts.append(experiment_count)
            both_counts.append(both_count)
            other_counts.append(other_count)
            total_counts.append(total_count)
            
            plenary_theory.append(plenary_theory_count)
            plenary_experiment.append(plenary_experiment_count)
            plenary_both.append(plenary_both_count)
            parallel_theory.append(parallel_theory_count)
            parallel_experiment.append(parallel_experiment_count)
            parallel_both.append(parallel_both_count)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Raw counts as stacked bars
    width = 0.8
    
    # Convert counts to percentages
    theory_pct = [100 * theory_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 for i in range(len(years))]
    experiment_pct = [100 * experiment_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 for i in range(len(years))]
    both_pct = [100 * both_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 for i in range(len(years))]
    other_pct = [100 * other_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 for i in range(len(years))]
    
    # Create the stacked bar chart
    bottom_vals = np.zeros(len(years))
    p1 = ax1.bar(years, theory_pct, width, label='Theory', color='#3498db', bottom=bottom_vals)
    bottom_vals = theory_pct
    p2 = ax1.bar(years, experiment_pct, width, label='Experiment', color='#e74c3c', bottom=bottom_vals)
    bottom_vals = [bottom_vals[i] + experiment_pct[i] for i in range(len(years))]
    p3 = ax1.bar(years, both_pct, width, label='Both', color='#9b59b6', bottom=bottom_vals)
    bottom_vals = [bottom_vals[i] + both_pct[i] for i in range(len(years))]
    p4 = ax1.bar(years, other_pct, width, label='Unclassified', color='#95a5a6', bottom=bottom_vals)
    
    # Add percentage labels on the bars
    for i, year in enumerate(years):
        if theory_pct[i] > 5:  # Only show label if segment is large enough
            ax1.text(year, theory_pct[i]/2, f"{theory_pct[i]:.0f}%", ha='center', va='center', color='white')
        if experiment_pct[i] > 5:
            ax1.text(year, theory_pct[i] + experiment_pct[i]/2, f"{experiment_pct[i]:.0f}%", ha='center', va='center', color='white')
    
    ax1.set_xlabel('Conference Year', fontsize=12)
    ax1.set_ylabel('Percentage of Presentations', fontsize=12)
    ax1.set_title('Theory vs Experiment Balance by Year', fontsize=14)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Theory/Experiment balance by presentation type
    width = 0.35
    x = np.arange(len(years))
    
    # Calculate ratios (ensuring we don't divide by zero)
    plenary_ratio = []
    parallel_ratio = []
    for i in range(len(years)):
        if plenary_experiment[i] > 0:
            plenary_ratio.append(plenary_theory[i] / plenary_experiment[i])
        else:
            plenary_ratio.append(0)
            
        if parallel_experiment[i] > 0:
            parallel_ratio.append(parallel_theory[i] / parallel_experiment[i])
        else:
            parallel_ratio.append(0)
    
    ax2.plot(years, plenary_ratio, 'o-', linewidth=2, label='Plenary', color='#e67e22')
    ax2.plot(years, parallel_ratio, 's-', linewidth=2, label='Parallel', color='#27ae60')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Equal balance')
    
    # Annotate the plot
    ax2.set_xlabel('Conference Year', fontsize=12)
    ax2.set_ylabel('Theory / Experiment Ratio', fontsize=12)
    ax2.set_title('Theory/Experiment Ratio by Presentation Type', fontsize=14)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add text annotations to explain
    ax2.text(0.05, 0.95, "Above 1: More theory presentations\nBelow 1: More experimental presentations", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust for legend at bottom
    plt.savefig('figures/theory_experiment_balance.pdf', bbox_inches='tight')
    plt.savefig('figures/theory_experiment_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'years': years,
        'theory_counts': theory_counts,
        'experiment_counts': experiment_counts,
        'both_counts': both_counts,
        'other_counts': other_counts,
        'total_counts': total_counts
    }
