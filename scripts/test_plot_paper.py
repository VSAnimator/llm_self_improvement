import pandas as pd
import matplotlib.pyplot as plt

# Function to plot data with customizable parameters
def plot_performance(csv_file, title=None, output_file=None, fig_width=4, fig_height=3, font_size=10):
    # Read CSV data
    df = pd.read_csv(csv_file)

    # Extract relevant columns
    tasks = df['Number of Training Tasks']
    
    # Plot average and any other non-Trial columns
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    # Create the plot
    ax = plt.gca()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot all non-Trial columns
    non_trial_cols = [col for col in df.columns if 'Trial' not in col and col != 'Number of Training Tasks']
    lines = []
    for col in non_trial_cols:
        line, = plt.plot(tasks, df[col], marker='o', linestyle='-', linewidth=2, label=col)
        lines.append(line)

    # Plot individual trials with lighter dashed lines
    trial_cols = [col for col in df.columns if 'Trial' in col]
    for trial in trial_cols:
        plt.plot(tasks, df[trial], linestyle='--', alpha=0.6, linewidth=1)

    # Customize labels and title with adjustable font size
    plt.xlabel('Number of Training Tasks', fontsize=font_size)
    plt.ylabel('Success Rate', fontsize=font_size)
    plt.title(title, fontsize=font_size+2)

    # Grid for readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize tick font sizes
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)

    # Generate output filename if not provided
    if output_file is None:
        output_file = f"{csv_file.split('.')[0]}_w{fig_width}_h{fig_height}_fs{font_size}.pdf"
    
    # Handle legend placement based on title
    if title and 'InterCode' in title:
        # Remove in-graph legend
        plt.legend().set_visible(False)
        
        # Create a separate figure for the vertical legend (stacked)
        figlegend_vertical = plt.figure(figsize=(2, 1))
        figlegend_vertical.legend(lines, [line.get_label() for line in lines], 
                        loc='center', ncol=1, fontsize=font_size-2)
        
        # Save the vertical legend as a separate file
        legend_output_vertical = f"{output_file}_legend_vertical.pdf"
        figlegend_vertical.savefig(legend_output_vertical, bbox_inches='tight')
        print(f"Vertical legend saved to {legend_output_vertical}")
        
        # Create a separate figure for the horizontal legend
        figlegend_horizontal = plt.figure(figsize=(4, 0.5))
        figlegend_horizontal.legend(lines, [line.get_label() for line in lines], 
                        loc='center', ncol=len(lines), fontsize=font_size-2)
        
        # Save the horizontal legend as a separate file
        legend_output_horizontal = f"{output_file}_legend_horizontal.pdf"
        figlegend_horizontal.savefig(legend_output_horizontal, bbox_inches='tight')
        print(f"Horizontal legend saved to {legend_output_horizontal}")
        
        # Adjust layout for the main figure
        plt.tight_layout()
    else:
        # For other plots, don't show any legend
        plt.legend().set_visible(False)
        plt.tight_layout()
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance plots from CSV data')
    parser.add_argument('--csv_file', help='Path to the CSV file containing performance data')
    parser.add_argument('--title', '-t', help='Title of the plot')
    parser.add_argument('--output', '-o', help='Output file name (default: auto-generated from parameters)')
    parser.add_argument('--width', '-w', type=float, default=4, help='Figure width in inches (default: 4)')
    parser.add_argument('--height', '-ht', type=float, default=3, help='Figure height in inches (default: 3)')
    parser.add_argument('--fontsize', '-f', type=int, default=17, help='Base font size (default: 10)')
    
    args = parser.parse_args()
    
    plot_performance(
        args.csv_file, 
        title=args.title,
        output_file=args.output,
        fig_width=args.width, 
        fig_height=args.height, 
        font_size=args.fontsize
    )