import os
import re
import json
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

class RetrievalAnalytics:
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        self.example_id_pattern = re.compile(r'Success entry ids: \[(.*?)\]')
        self.failure_id_pattern = re.compile(r'Failure entry ids: \[(.*?)\]')
        self.reward_pattern = re.compile(r'Reward: (\d+)')
        self.goal_pattern = re.compile(r'Goal: Your task is to: (.*?)(?:\n|$)')
        
        # Data structures
        self.example_tree = nx.DiGraph()  # Tracks which examples led to which new examples
        self.example_success_count = Counter()  # How many times each example was used in successful tasks
        self.example_failure_count = Counter()  # How many times each example was used in failed tasks
        self.example_usage_count = Counter()    # How many times each example was retrieved
        self.example_task_types = defaultdict(Counter)  # Maps examples to the tasks they helped with
        
        # Initialize with base examples
        for i in range(1, 19):  # Initial 18 examples
            self.example_tree.add_node(i, type="base", success_rate=0, usage_count=0)
    
    def parse_log_file(self, file_path, example_id):
        """Parse a single log file and extract relevant information."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract goal
        goal_match = self.goal_pattern.search(content)
        goal = goal_match.group(1) if goal_match else "Unknown"
        
        # Extract success and failure entry IDs throughout the file
        success_matches = self.example_id_pattern.findall(content)
        failure_matches = self.failure_id_pattern.findall(content)
        
        # Extract final reward to determine if task was successful
        rewards = self.reward_pattern.findall(content)
        final_reward = int(rewards[-1]) if rewards else 0
        is_successful = final_reward > 0
        
        # Process retrieved examples
        all_retrieved_ids = set()
        for match in success_matches:
            if match.strip():
                ids = [int(id_str.strip()) for id_str in match.split(',')]
                all_retrieved_ids.update(ids)
                
                # Update individual example statistics
                for retrieved_id in ids:
                    self.example_usage_count[retrieved_id] += 1/len(success_matches)
                    if is_successful:
                        self.example_success_count[retrieved_id] += 1/len(success_matches)
                    else:
                        self.example_failure_count[retrieved_id] += 1/len(success_matches)
                    self.example_task_types[retrieved_id][goal] += 1/len(success_matches)
        
        # Track relationships in the example tree
        for retrieved_id in all_retrieved_ids:
            if retrieved_id not in self.example_tree:
                self.example_tree.add_node(retrieved_id, type="generated", success_rate=0, usage_count=0)
            self.example_tree.add_edge(retrieved_id, example_id)
        
        return {
            "example_id": example_id,
            "goal": goal,
            "retrieved_ids": list(all_retrieved_ids),
            "is_successful": is_successful,
            "final_reward": final_reward
        }
    
    def analyze_logs(self):
        """Process all log files in the directory."""
        log_files = sorted([f for f in os.listdir(self.logs_dir) if f.endswith('.txt')], 
                          key=lambda x: int(x.split('.')[0]))
        
        results = []
        for file_name in tqdm(log_files, desc="Processing log files"):
            file_path = os.path.join(self.logs_dir, file_name)
            example_id = 18 + int(file_name.split('.')[0]) + 1  # Base 18 examples + file number + 1
            
            # Parse the log file
            log_result = self.parse_log_file(file_path, example_id)
            results.append(log_result)
            
            if example_id not in self.example_tree:
                self.example_tree.add_node(example_id, 
                                         type="generated", 
                                         success_rate=int(log_result["is_successful"]),
                                         usage_count=0)
        
        # Update node attributes with success rates
        for node in self.example_tree.nodes():
            usage = self.example_usage_count[node]
            success = self.example_success_count[node]
            if usage > 0:
                print(f"Updating node {node} with success rate {success / usage}", "usage", usage, "success", success)
                self.example_tree.nodes[node]['success_rate'] = success / usage
                self.example_tree.nodes[node]['usage_count'] = usage
        
        return results
    
    def calculate_cascade_influence(self):
        """Calculate the cascade influence of each example."""
        influence_scores = {}
        
        for node in self.example_tree.nodes():
            if 'success_rate' not in self.example_tree.nodes[node]:
                self.example_tree.nodes[node]['success_rate'] = 0
                self.example_tree.nodes[node]['usage_count'] = 0
            # Direct influence: how well this example performs when retrieved
            direct_success_rate = self.example_tree.nodes[node]['success_rate']
            usage_count = self.example_tree.nodes[node]['usage_count']
            
            # Cascade influence: how well examples derived from this one perform
            descendants = nx.descendants(self.example_tree, node)
            if descendants:
                descendant_success_rates = [self.example_tree.nodes[d]['success_rate'] 
                                           for d in descendants
                                           if 'usage_count' in self.example_tree.nodes[d] and self.example_tree.nodes[d]['usage_count'] > 0]
                cascade_rate = np.mean(descendant_success_rates) if descendant_success_rates else 0
            else:
                cascade_rate = 0
            
            # Combined score
            if usage_count > 0:
                influence_scores[node] = {
                    'direct_success_rate': direct_success_rate,
                    'cascade_success_rate': cascade_rate,
                    'combined_score': (direct_success_rate + cascade_rate) / 2,
                    'usage_count': usage_count,
                    'descendant_count': len(descendants)
                }
        
        return influence_scores
    
    def generate_reports(self):
        """Generate various analysis reports."""
        # Calculate example statistics
        example_stats = {}
        for example_id in self.example_usage_count.keys():
            usage = self.example_usage_count[example_id]
            success = self.example_success_count[example_id]
            failure = self.example_failure_count[example_id]
            
            if usage > 0:
                success_rate = success / usage
            else:
                success_rate = 0
                
            example_stats[example_id] = {
                'usage_count': usage,
                'success_count': success,
                'failure_count': failure,
                'success_rate': success_rate,
                'task_types': dict(self.example_task_types[example_id])
            }
        
        # Calculate cascade influence
        influence_scores = self.calculate_cascade_influence()
        
        return {
            'example_stats': example_stats,
            'influence_scores': influence_scores
        }
    
    def visualize_example_tree(self, output_path="example_tree.png", min_usage=1):
        """Visualize the example tree with node sizes based on usage and colors based on success rate."""
        # Create a copy of the graph to modify for visualization
        G = self.example_tree.copy()
        
        # Remove nodes with usage below threshold
        nodes_to_remove = [n for n, attr in G.nodes(data=True) if attr['usage_count'] < min_usage]
        G.remove_nodes_from(nodes_to_remove)
        
        # Prepare node colors based on success rate (green for high success, red for low)
        node_colors = []
        for node in G.nodes():
            success_rate = G.nodes[node]['success_rate']
            # RGB: Red (low success) to Green (high success)
            color = (1-success_rate, success_rate, 0)
            node_colors.append(color)
        
        # Prepare node sizes based on usage count
        node_sizes = [50 + 10 * G.nodes[node]['usage_count'] for node in G.nodes()]
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges with reduced alpha for clarity
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Example Influence Tree (Node Size = Usage, Color = Success Rate)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_top_performers(self, output_path="top_performers.png", top_n=20):
        """Visualize the top-performing examples based on combined influence score."""
        influence_scores = self.calculate_cascade_influence()
        
        # Sort examples by combined score
        sorted_examples = sorted(influence_scores.items(), 
                                key=lambda x: x[1]['combined_score'], 
                                reverse=True)[:top_n]
        
        example_ids = [s[0] for s in sorted_examples]
        direct_scores = [s[1]['direct_success_rate'] for s in sorted_examples]
        cascade_scores = [s[1]['cascade_success_rate'] for s in sorted_examples]
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.8
        
        plt.bar(example_ids, direct_scores, bar_width, label='Direct Success Rate')
        plt.bar(example_ids, cascade_scores, bar_width, bottom=direct_scores, label='Cascade Success Rate')
        
        plt.xlabel('Example ID')
        plt.ylabel('Success Rate')
        plt.title(f'Top {top_n} Most Influential Examples')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def run_analysis(self, output_dir="results"):
        """Run the full analysis pipeline."""
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process logs
        print("Analyzing log files...")
        self.analyze_logs()
        
        # Generate reports
        print("Generating reports...")
        reports = self.generate_reports()
        
        # Save reports to disk
        with open(os.path.join(output_dir, "example_stats.json"), 'w') as f:
            json.dump(reports['example_stats'], f, indent=2)
        
        with open(os.path.join(output_dir, "influence_scores.json"), 'w') as f:
            json.dump(reports['influence_scores'], f, indent=2)
        
        # Create summary file
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write("RETRIEVAL ANALYSIS SUMMARY\n")
            f.write("=========================\n\n")
            
            # Overall statistics
            total_examples = len(self.example_usage_count)
            f.write(f"Total examples analyzed: {total_examples}\n")
            
            # Success rates
            avg_success_rate = sum(ex['success_rate'] for ex in reports['example_stats'].values()) / total_examples
            f.write(f"Average example success rate: {avg_success_rate:.2f}\n\n")
            
            # Top 10 most used examples
            f.write("Top 10 most frequently retrieved examples:\n")
            top_usage = sorted(reports['example_stats'].items(), key=lambda x: x[1]['usage_count'], reverse=True)[:10]
            for ex_id, stats in top_usage:
                f.write(f"  Example {ex_id}: Used {stats['usage_count']} times, {stats['success_rate']:.2f} success rate\n")
            f.write("\n")
            
            # Top 10 most successful examples (min 5 usages)
            f.write("Top 10 most successful examples (min 5 uses):\n")
            top_success = sorted(
                [item for item in reports['example_stats'].items() if item[1]['usage_count'] >= 5],
                key=lambda x: x[1]['success_rate'], 
                reverse=True
            )[:10]
            for ex_id, stats in top_success:
                f.write(f"  Example {ex_id}: {stats['success_rate']:.2f} success rate, used {stats['usage_count']} times\n")
            f.write("\n")
            
            # Top 10 most influential examples
            f.write("Top 10 most influential examples (combined score):\n")
            top_influence = sorted(reports['influence_scores'].items(), 
                                  key=lambda x: x[1]['combined_score'], 
                                  reverse=True)[:10]
            for ex_id, scores in top_influence:
                f.write(f"  Example {ex_id}: Combined score {scores['combined_score']:.2f}, "
                        f"direct {scores['direct_success_rate']:.2f}, "
                        f"cascade {scores['cascade_success_rate']:.2f}\n")
        
        # Generate visualizations
        print("Generating visualizations...")
        #self.visualize_example_tree(os.path.join(output_dir, "example_tree.png"))
        #self.visualize_example_tree(os.path.join(output_dir, "significant_examples_tree.png"), min_usage=5)
        #self.visualize_top_performers(os.path.join(output_dir, "top_performers.png"))
        
        print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze LLM agent retrieval logs')
    parser.add_argument('logs_dir', help='Directory containing log files')
    parser.add_argument('--output', default='results', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = RetrievalAnalytics(args.logs_dir)
    analyzer.run_analysis(args.output)