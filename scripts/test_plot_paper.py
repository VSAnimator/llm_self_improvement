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
    
    # Plot all non-Trial columns
    non_trial_cols = [col for col in df.columns if 'Trial' not in col and col != 'Number of Training Tasks']
    for col in non_trial_cols:
        plt.plot(tasks, df[col], marker='o', linestyle='-', linewidth=2, label=col)

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

    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = f"{csv_file.split('.')[0]}_w{fig_width}_h{fig_height}_fs{font_size}.pdf"
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance plots from CSV data')
    parser.add_argument('--csv_file', help='Path to the CSV file containing performance data')
    parser.add_argument('--title', '-t', help='Title of the plot')
    parser.add_argument('--output', '-o', help='Output file name (default: auto-generated from parameters)')
    parser.add_argument('--width', '-w', type=float, default=4, help='Figure width in inches (default: 4)')
    parser.add_argument('--height', '-ht', type=float, default=3, help='Figure height in inches (default: 3)')
    parser.add_argument('--fontsize', '-f', type=int, default=9, help='Base font size (default: 10)')
    
    args = parser.parse_args()
    
    plot_performance(
        args.csv_file, 
        title=args.title,
        output_file=args.output,
        fig_width=args.width, 
        fig_height=args.height, 
        font_size=args.fontsize
    )
