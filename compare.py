import subprocess
import argparse
import os
import pandas as pd
import re
from tabulate import tabulate
import matplotlib.pyplot as plt

def run_profile(kernel_index, matrix_size=4096, num_runs=1):
    """Run the NCU profiler for a specific kernel and return the raw text results."""
    print(f"Profiling kernel {kernel_index}...")
    
    # Create output directory if it doesn't exist
    os.makedirs("profile_results", exist_ok=True)
    
    # Create output filename
    output_file = f"profile_results/kernel_{kernel_index}_profile.txt"
    
    # Run the ncu command with output to a text file
    cmd = [
        "sudo", "/usr/local/cuda-12.4/bin/ncu", 
        "--set", "full", 
        "--log-file", output_file,
        "./build/sgemm_runner", 
        str(matrix_size), str(matrix_size), str(matrix_size), 
        str(num_runs), str(kernel_index), "1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Read the output file
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                return f.read()
        else:
            print(f"Output file {output_file} not found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running profiler: {e}")
        return None

def parse_ncu_output(profile_text):
    """Parse NCU output text into sections with tables."""
    if not profile_text:
        return {}
    
    # Dictionary to hold all section tables
    section_tables = {}
    
    # Split by section
    section_pattern = r'Section: ([^\n]+)'
    sections = re.split(section_pattern, profile_text)
    
    # Skip the first element which is text before the first section
    for i in range(1, len(sections), 2):
        if i+1 >= len(sections):
            break
            
        section_name = sections[i].strip()
        section_content = sections[i+1].strip()
        
        # Look for tables in the section
        # Tables typically have a header row and data rows separated by dashed lines
        table_data = parse_table(section_content)
        
        if table_data is not None and not table_data.empty:
            section_tables[section_name] = table_data
    
    return section_tables

def parse_table(section_text):
    """Parse a table from section text using fixed-width column detection."""
    lines = section_text.strip().split('\n')
    
    # Find separator lines (consisting of dashes)
    separator_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^[\s-]+$', line):
            separator_indices.append(i)
    
    # Need at least two separator lines to identify a table
    if len(separator_indices) < 2:
        return None
    
    # Get header line (between first two separator lines)
    if separator_indices[0] + 1 < separator_indices[1]:
        header_line = lines[separator_indices[0] + 1]
    else:
        return None
    
    # Use first separator line to determine column widths
    separator_line = lines[separator_indices[0]]
    
    # Find column boundaries by looking at groups of dashes
    col_boundaries = []
    in_dashes = False
    for i, char in enumerate(separator_line):
        if char == '-' and not in_dashes:
            col_boundaries.append(i)  # Start of column
            in_dashes = True
        elif char != '-' and in_dashes:
            col_boundaries.append(i)  # End of column
            in_dashes = False
    
    # If the line ends with dashes, add the end position
    if in_dashes:
        col_boundaries.append(len(separator_line))
    
    # Must have an even number of boundaries (start and end for each column)
    if len(col_boundaries) % 2 != 0:
        return None
        
    # Extract column names from header line
    column_names = []
    for i in range(0, len(col_boundaries), 2):
        start = col_boundaries[i]
        end = col_boundaries[i+1] if i+1 < len(col_boundaries) else len(header_line)
        column_names.append(header_line[start:end].strip())
    
    # Extract data rows (between second separator and last separator)
    data_rows = []
    for i in range(separator_indices[1] + 1, separator_indices[-1]):
        line = lines[i]
        # Skip empty lines or information lines starting with INF/OPT
        if not line.strip() or line.strip().startswith(('INF', 'OPT')):
            continue
            
        # Extract column values using the same boundaries
        row_values = []
        for j in range(0, len(col_boundaries), 2):
            start = col_boundaries[j]
            end = col_boundaries[j+1] if j+1 < len(col_boundaries) else len(line)
            # Ensure we're not going past the line length
            if start < len(line):
                value = line[start:min(end, len(line))].strip()
                row_values.append(value)
            else:
                row_values.append('')
                
        if row_values:
            data_rows.append(row_values)
    
    # Create DataFrame
    if column_names and data_rows:
        try:
            df = pd.DataFrame(data_rows, columns=column_names)
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None
    
    return None

def compare_sections(kernel_indices, parsed_data_list):
    """Compare tables from each section between kernels."""
    # Get the set of all section names
    all_sections = set()
    for parsed_data in parsed_data_list:
        all_sections.update(parsed_data.keys())
    
    # For each section, compare the tables
    for section_name in sorted(all_sections):
        print(f"\n{'=' * 20} SECTION: {section_name} {'=' * 20}\n")
        
        # Collect metrics from all kernels for this section
        metrics_data = {}
        
        for i, parsed_data in enumerate(parsed_data_list):
            kernel_name = f"Kernel {kernel_indices[i]}"
            
            if section_name in parsed_data and parsed_data[section_name] is not None:
                df = parsed_data[section_name]
                
                # For typical NCU format with Metric Name/Value columns
                if 'Metric Name' in df.columns and 'Metric Value' in df.columns:
                    for _, row in df.iterrows():
                        metric_name = row['Metric Name']
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = {}
                        metrics_data[metric_name][kernel_name] = row['Metric Value']
                else:
                    # For other table formats, use the first column as index
                    first_col = df.columns[0]
                    for _, row in df.iterrows():
                        index_val = row[first_col]
                        if index_val not in metrics_data:
                            metrics_data[index_val] = {}
                        
                        # Add other columns as values
                        for col in df.columns[1:]:
                            metrics_data[index_val][f"{kernel_name} {col}"] = row[col]
        
        if not metrics_data:
            print("No comparable metrics found in this section")
            continue
        
        # Create comparison DataFrame
        compare_df = pd.DataFrame.from_dict(metrics_data, orient='index')
        
        # Add difference column if comparing exactly two kernels and we have Metric Value format
        if len(kernel_indices) == 2 and all(f"Kernel {idx}" in compare_df.columns for idx in kernel_indices):
            kernel1 = f"Kernel {kernel_indices[0]}"
            kernel2 = f"Kernel {kernel_indices[1]}"
            
            compare_df['Difference (%)'] = ""
            
            for metric in compare_df.index:
                try:
                    val1 = convert_to_float(compare_df.loc[metric, kernel1])
                    val2 = convert_to_float(compare_df.loc[metric, kernel2])
                    
                    if val1 is not None and val2 is not None and val1 != 0:
                        diff_pct = (val2 - val1) / val1 * 100
                        compare_df.loc[metric, 'Difference (%)'] = f"{diff_pct:.2f}%"
                except Exception as e:
                    # Skip metrics that can't be compared
                    pass
        
        # Print the comparison table
        print(tabulate(compare_df, headers='keys', tablefmt='grid'))
        
        # Create visualization for key performance metrics
        if section_name == "GPU Speed Of Light Throughput":
            plot_performance_comparison(compare_df, kernel_indices)

def convert_to_float(value):
    """Convert string value to float, handling various formats."""
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
        
    # Remove non-numeric characters except for decimal point
    clean_value = re.sub(r'[^0-9.-]', '', value)
    
    try:
        return float(clean_value)
    except (ValueError, TypeError):
        return None

def plot_performance_comparison(df, kernel_indices):
    """Create a bar chart comparing key performance metrics."""
    # Key metrics to visualize - adjust based on what's in your data
    key_metrics = [
        'Memory Throughput',
        'DRAM Throughput',
        'L1/TEX Cache Throughput',
        'L2 Cache Throughput',
        'Compute (SM) Throughput'
    ]
    
    # Filter metrics that are actually in the dataframe
    available_metrics = [m for m in key_metrics if m in df.index]
    
    if not available_metrics:
        print("No key performance metrics available for visualization")
        return
    
    # Extract data for chart
    plot_data = {}
    kernel_columns = [f"Kernel {idx}" for idx in kernel_indices]
    
    for metric in available_metrics:
        if metric in df.index:
            plot_data[metric] = []
            for col in kernel_columns:
                if col in df.columns:
                    value = convert_to_float(df.loc[metric, col])
                    plot_data[metric].append(value if value is not None else 0)
                else:
                    plot_data[metric].append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    bar_width = 0.15
    index = range(len(available_metrics))
    
    for i, kernel_idx in enumerate(kernel_indices):
        values = [plot_data[metric][i] for metric in available_metrics]
        ax.bar([p + i * bar_width for p in index], values, bar_width, 
               label=f'Kernel {kernel_idx}')
    
    # Add labels and legend
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Comparison')
    ax.set_xticks([p + bar_width * (len(kernel_indices) - 1) / 2 for p in index])
    ax.set_xticklabels(available_metrics, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('profile_results/performance_comparison.png')
    print("\nPerformance comparison chart saved to 'profile_results/performance_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description="Profile and compare CUDA kernels")
    parser.add_argument("kernel_indices", type=int, nargs="+", help="Indices of kernels to profile and compare")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (default: 4096)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument("--use-existing", action="store_true", help="Use existing profile files instead of running NCU again")
    
    args = parser.parse_args()
    
    # Run profiling for each kernel or use existing profiles
    profile_texts = []
    
    for idx in args.kernel_indices:
        profile_file = f"profile_results/kernel_{idx}_profile.txt"
        
        if args.use_existing and os.path.exists(profile_file):
            print(f"Using existing profile for kernel {idx}...")
            with open(profile_file, 'r') as f:
                profile_texts.append(f.read())
        else:
            profile_text = run_profile(idx, args.size, args.runs)
            if profile_text:
                profile_texts.append(profile_text)
    
    # Parse each profile text
    parsed_data_list = []
    for profile_text in profile_texts:
        parsed_data = parse_ncu_output(profile_text)
        parsed_data_list.append(parsed_data)
    
    # Compare sections
    compare_sections(args.kernel_indices, parsed_data_list)

if __name__ == "__main__":
    main()


"""
python compare.py 3 4 --size 4096 --runs 1 --use-existing
"""