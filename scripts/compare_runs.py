import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path


class RetrievalRunComparison:
    def __init__(self, base_dir, run_folders, output_dir="comparison_results"):
        """
        Initialize the comparison tool.
        
        Args:
            base_dir (str): Base directory containing all run folders
            run_folders (list): List of folder names to compare
            output_dir (str): Directory to save comparison results
        """
        self.base_dir = Path(base_dir)
        self.run_folders = run_folders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data structures to hold results
        self.influence_data = {}
        self.example_stats = {}
        self.run_params = {}
        
    def load_run_data(self):
        """Load influence scores and other data from all specified runs."""
        print("Loading data from all runs...")
        
        for run_folder in self.run_folders:
            run_path = self.base_dir / run_folder
            influence_path = run_path / "influence_scores.json"
            stats_path = run_path / "example_stats.json"
            
            # Try to load run parameters if available
            params_path = run_path / "params.json"
            if params_path.exists():
                with open(params_path, 'r') as f:
                    self.run_params[run_folder] = json.load(f)
            else:
                self.run_params[run_folder] = {"run_name": run_folder}
            
            # Load influence scores
            if influence_path.exists():
                with open(influence_path, 'r') as f:
                    self.influence_data[run_folder] = json.load(f)
                print(f"Loaded influence data from {run_folder}")
            else:
                print(f"Warning: No influence scores found for {run_folder}")
                
            # Load example stats
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.example_stats[run_folder] = json.load(f)
                print(f"Loaded example stats from {run_folder}")
            else:
                print(f"Warning: No example stats found for {run_folder}")
    
    def create_combined_dataframe(self):
        """Create a combined DataFrame of influence scores across all runs."""
        print("Creating combined dataframe...")
        
        # Prepare data for DataFrame
        all_data = []
        
        for run_name, influence_scores in self.influence_data.items():
            for example_id, scores in influence_scores.items():
                entry = {
                    'run': run_name,
                    'example_id': int(example_id),
                    'direct_success_rate': scores['direct_success_rate'],
                    'cascade_success_rate': scores['cascade_success_rate'],
                    'combined_score': scores['combined_score'],
                    'usage_count': scores['usage_count'],
                    'descendant_count': scores['descendant_count']
                }
                
                # Add run parameters if available
                if run_name in self.run_params:
                    for param, value in self.run_params[run_name].items():
                        entry[f'param_{param}'] = value
                
                all_data.append(entry)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Save the combined data
        df.to_csv(self.output_dir / "combined_influence_data.csv", index=False)
        
        return df
    
    def find_differential_performance(self, min_usage=5, diff_threshold=0.3):
        """
        Find examples with significantly different performance across runs.
        
        Args:
            min_usage (int): Minimum number of times an example must be used to be considered
            diff_threshold (float): Minimum difference in success rate to be considered significant
        
        Returns:
            DataFrame: Examples with differential performance
        """
        print(f"Finding examples with differential performance (threshold={diff_threshold})...")
        
        df = self.create_combined_dataframe()
        
        # Filter by minimum usage
        df_filtered = df[df['usage_count'] >= min_usage]
        
        # Find examples used in multiple runs
        example_counts = df_filtered['example_id'].value_counts()
        multi_run_examples = example_counts[example_counts > 1].index.tolist()
        
        # Calculate performance differences for each example across runs
        diff_results = []
        
        for example_id in multi_run_examples:
            example_df = df_filtered[df_filtered['example_id'] == example_id]
            
            # Calculate stats
            max_direct = example_df['direct_success_rate'].max()
            min_direct = example_df['direct_success_rate'].min()
            max_cascade = example_df['cascade_success_rate'].max()
            min_cascade = example_df['cascade_success_rate'].min()
            max_combined = example_df['combined_score'].max()
            min_combined = example_df['combined_score'].min()
            
            direct_diff = max_direct - min_direct
            cascade_diff = max_cascade - min_cascade
            combined_diff = max_combined - min_combined
            
            # If any difference is above threshold, include this example
            if (direct_diff >= diff_threshold or 
                cascade_diff >= diff_threshold or 
                combined_diff >= diff_threshold):
                
                best_run = example_df.loc[example_df['direct_success_rate'].idxmax(), 'run']
                worst_run = example_df.loc[example_df['direct_success_rate'].idxmin(), 'run']
                
                diff_results.append({
                    'example_id': example_id,
                    'direct_diff': direct_diff,
                    'cascade_diff': cascade_diff,
                    'combined_diff': combined_diff,
                    'best_run': best_run,
                    'worst_run': worst_run,
                    'max_direct': max_direct,
                    'min_direct': min_direct,
                    'run_count': len(example_df),
                    'avg_usage': example_df['usage_count'].mean()
                })
        
        # Convert to DataFrame and sort by largest difference
        diff_df = pd.DataFrame(diff_results)
        if not diff_df.empty:
            diff_df = diff_df.sort_values('direct_diff', ascending=False)
            
            # Save results
            diff_df.to_csv(self.output_dir / "differential_performance.csv", index=False)
        else:
            print("No examples with differential performance found.")
            
        return diff_df
    
    def analyze_run_differences(self):
        """Analyze what parameters might cause performance differences."""
        print("Analyzing parameter differences between runs...")
        
        # Get differential performance examples
        diff_df = self.find_differential_performance()
        
        if diff_df.empty:
            print("No differential examples found. Cannot analyze run differences.")
            return None
        
        # Create a DataFrame to hold parameter differences
        param_analysis = []
        
        # For each differential example, analyze what parameters differ between best and worst runs
        for _, row in diff_df.iterrows():
            example_id = row['example_id']
            best_run = row['best_run']
            worst_run = row['worst_run']
            
            # Skip if either run doesn't have parameters
            if best_run not in self.run_params or worst_run not in self.run_params:
                continue
                
            best_params = self.run_params[best_run]
            worst_params = self.run_params[worst_run]
            
            # Find differences in parameters
            diff_params = {}
            all_params = set(best_params.keys()) | set(worst_params.keys())
            
            for param in all_params:
                best_value = best_params.get(param, "N/A")
                worst_value = worst_params.get(param, "N/A")
                
                if best_value != worst_value:
                    diff_params[param] = f"{best_value} vs {worst_value}"
            
            param_analysis.append({
                'example_id': example_id,
                'direct_diff': row['direct_diff'],
                'best_run': best_run,
                'worst_run': worst_run,
                'different_params': diff_params
            })
        
        param_df = pd.DataFrame(param_analysis)
        
        if not param_df.empty:
            # Save results
            param_df.to_json(self.output_dir / "parameter_differences.json", orient='records', indent=2)
            
            # Count which parameters most often differ in high-differential examples
            param_counts = defaultdict(int)
            for diff_params in param_df['different_params']:
                for param in diff_params:
                    param_counts[param] += 1
            
            with open(self.output_dir / "parameter_frequency.txt", 'w') as f:
                f.write("Parameters that most frequently differ between best and worst runs:\n")
                f.write("=================================================================\n\n")
                
                for param, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{param}: {count} examples\n")
        
        return param_df
    
    def visualize_example_performance(self, top_n=10):
        """
        Visualize performance differences of the top examples across runs.
        
        Args:
            top_n (int): Number of top differential examples to visualize
        """
        print(f"Visualizing top {top_n} examples with differential performance...")
        
        # Get differential performance data
        diff_df = self.find_differential_performance()
        
        if diff_df.empty:
            print("No differential examples found. Cannot create visualizations.")
            return
        
        # Get combined data
        combined_df = pd.read_csv(self.output_dir / "combined_influence_data.csv")
        
        # Get top examples with largest direct success rate difference
        top_examples = diff_df.nlargest(top_n, 'direct_diff')['example_id'].tolist()
        
        # Plot direct success rate for each example across runs
        for example_id in top_examples:
            example_data = combined_df[combined_df['example_id'] == example_id].copy()
            example_data = example_data.sort_values('direct_success_rate')
            
            plt.figure(figsize=(10, 6))
            
            # Create a bar chart
            bars = plt.bar(example_data['run'], example_data['direct_success_rate'], alpha=0.7)
            
            # Add usage count as text on each bar
            for i, bar in enumerate(bars):
                usage = example_data.iloc[i]['usage_count']
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    0.05, 
                    f"Used: {int(usage)}", 
                    ha='center', 
                    rotation=90, 
                    color='black'
                )
            
            plt.title(f"Example {example_id} Performance Across Runs")
            plt.xlabel("Run")
            plt.ylabel("Direct Success Rate")
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / f"example_{example_id}_comparison.png", dpi=300)
            plt.close()
        
        # Create a heatmap of the top examples across runs
        self._create_performance_heatmap(combined_df, top_examples)
    
    def _create_performance_heatmap(self, df, example_ids):
        """
        Create a heatmap showing performance of selected examples across runs.
        
        Args:
            df (DataFrame): Combined data
            example_ids (list): List of example IDs to include
        """
        # Filter data to only selected examples
        filtered_df = df[df['example_id'].isin(example_ids)]
        
        # Create a pivot table: runs x examples with direct success rate as values
        pivot_df = filtered_df.pivot_table(
            index='run', 
            columns='example_id', 
            values='direct_success_rate',
            aggfunc='mean'
        )
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
        
        plt.title("Direct Success Rate of Top Differential Examples Across Runs")
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_examples_heatmap.png", dpi=300)
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        print("Generating comparison report...")
        
        # Get differential performance data
        diff_df = self.find_differential_performance()
        
        if diff_df.empty:
            print("No differential examples found. Creating minimal report.")
        
        # Create report
        with open(self.output_dir / "comparison_report.md", 'w') as f:
            f.write("# Retrieval Run Comparison Report\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"This report compares example performance across {len(self.run_folders)} different runs.\n\n")
            f.write(f"Runs analyzed: {', '.join(self.run_folders)}\n\n")
            
            # Run parameters
            f.write("## Run Parameters\n\n")
            for run, params in self.run_params.items():
                f.write(f"### {run}\n\n")
                for param, value in params.items():
                    f.write(f"- **{param}**: {value}\n")
                f.write("\n")
            
            # Examples with differential performance
            if not diff_df.empty:
                f.write("## Examples with Significant Performance Differences\n\n")
                f.write("The following examples showed significantly different performance across runs:\n\n")
                
                f.write("| Example ID | Direct Diff | Best Run | Worst Run | Best Rate | Worst Rate |\n")
                f.write("|------------|------------|----------|-----------|-----------|------------|\n")
                
                for _, row in diff_df.iterrows():
                    f.write(f"| {int(row['example_id'])} | {row['direct_diff']:.3f} | {row['best_run']} | {row['worst_run']} | {row['max_direct']:.3f} | {row['min_direct']:.3f} |\n")
                
                f.write("\n")
                
                # Include visualizations
                f.write("## Visualizations\n\n")
                f.write("Individual performance charts were generated for the top examples with differential performance.\n\n")
                f.write("![Heatmap](top_examples_heatmap.png)\n\n")
            else:
                f.write("## No Significant Performance Differences\n\n")
                f.write("No examples showed significant performance differences across the analyzed runs.\n\n")
            
            # Parameter analysis summary
            param_analysis_file = self.output_dir / "parameter_differences.json"
            if param_analysis_file.exists():
                f.write("## Parameter Analysis\n\n")
                f.write("The following parameters most frequently differed between best and worst runs:\n\n")
                
                with open(self.output_dir / "parameter_frequency.txt", 'r') as param_file:
                    param_content = param_file.read()
                    # Skip the header lines
                    param_content = "\n".join(param_content.split("\n")[3:])
                    f.write("```\n" + param_content + "\n```\n\n")
                
                f.write("This suggests these parameters may have the most impact on example effectiveness.\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            if not diff_df.empty:
                top_example = diff_df.iloc[0]['example_id']
                f.write(f"Example {int(top_example)} showed the largest performance difference across runs, ")
                f.write(f"with a direct success rate difference of {diff_df.iloc[0]['direct_diff']:.3f}.\n\n")
                f.write("Consider analyzing these high-differential examples more closely to understand why their performance varies so significantly across different configurations.\n")
            else:
                f.write("Examples showed consistent performance across the analyzed runs. This suggests that the effectiveness of examples is relatively stable across the tested hyperparameter configurations.\n")
    
    def find_best_examples_per_task(self, max_task_id=1200):
        """
        For each task ID up to max_task_id, find the example that achieves the highest direct success rate.
        
        Args:
            max_task_id (int): Maximum task ID to analyze (default: 200)
            
        Returns:
            dict: Dictionary mapping task IDs to their best examples across all runs
        """
        print(f"Finding best examples for the first {max_task_id} tasks...")
        
        # First, load example stats if not already loaded
        if not self.example_stats:
            self.load_run_data()
        
        best_examples = {}
        worst_examples = {}
        
        # For each task ID up to max_task_id
        for task_id in range(1, max_task_id + 1):
            best_success_rate = 0
            best_example = None
            best_run = None
            worst_success_rate = 1
            worst_example = None
            worst_run = None
            
            # Check each run
            for run_name, stats in self.example_stats.items():
                # Check each example in this run
                if str(task_id) not in stats:
                    continue
                example_data = stats[str(task_id)]
                success_rate = example_data['success_rate']
                
                # If this example has a better success rate for this task, update our best
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_example = str(task_id)
                    best_run = run_name

                # If this example has a worse success rate for this task, update our worst
                if success_rate < worst_success_rate:
                    worst_success_rate = success_rate
                    worst_example = str(task_id)
                    worst_run = run_name
            
            # If we found a best example for this task, add it to our results
            if best_example is not None:
                best_examples[task_id] = {
                    "run": best_run,
                    "success_rate": best_success_rate
                }

            # If we found a worst example for this task, add it to our results
            if worst_example is not None:
                worst_examples[task_id] = {
                    "run": worst_run,
                    "success_rate": worst_success_rate
                }
        
        # Save the results to a file
        output_file = self.output_dir / "best_examples_per_task.json"
        with open(output_file, 'w') as f:
            json.dump(best_examples, f, indent=2)

        output_file = self.output_dir / "worst_examples_per_task.json"
        with open(output_file, 'w') as f:
            json.dump(worst_examples, f, indent=2)
        
        print(f"Best examples for {len(best_examples)} tasks saved to {output_file}")
        
        # Also create a CSV version for easier analysis
        df_rows = []
        for task_id, data in best_examples.items():
            df_rows.append({
                "task_id": task_id,
                "best_run": data["run"],
                "success_rate": data["success_rate"]
            })
        
        df = pd.DataFrame(df_rows)
        csv_file = self.output_dir / "best_examples_per_task.csv"
        df.to_csv(csv_file, index=False)

        df_rows = []
        for task_id, data in worst_examples.items():
            df_rows.append({
                "task_id": task_id,
                "worst_run": data["run"],
                "success_rate": data["success_rate"]
            })

        df = pd.DataFrame(df_rows)
        csv_file = self.output_dir / "worst_examples_per_task.csv"
        df.to_csv(csv_file, index=False)
        
        return best_examples

    def run_comparison(self):
        """Run the full comparison analysis pipeline."""
        print(f"Starting comparison of {len(self.run_folders)} runs...")
        
        # Load data from all runs
        self.load_run_data()
        
        # Create combined dataframe
        self.create_combined_dataframe()
        
        # Find examples with differential performance
        self.find_differential_performance()
        
        # Analyze run differences
        self.analyze_run_differences()
        
        # Visualize example performance
        self.visualize_example_performance()
        
        # Generate report
        self.generate_report()

        # Find best examples per task
        self.find_best_examples_per_task()
        
        print(f"Comparison complete. Results saved to {self.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare influence scores across multiple retrieval runs')
    parser.add_argument('base_dir', help='Base directory containing all run folders')
    parser.add_argument('--runs', nargs='+', required=True, help='List of run folder names to compare')
    parser.add_argument('--output', default='comparison_results', help='Output directory for comparison results')
    parser.add_argument('--min-usage', type=int, default=5, help='Minimum usage count to consider an example')
    parser.add_argument('--diff-threshold', type=float, default=0.3, help='Minimum difference in success rate to be significant')
    
    args = parser.parse_args()
    
    comparator = RetrievalRunComparison(args.base_dir, args.runs, args.output)
    comparator.run_comparison()