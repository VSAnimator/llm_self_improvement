import os
import re
import shutil
from collections import defaultdict

"""
log_path = "pbt_6ic_seg20_log.txt"
log_root = "logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_6ic_seg20"
output_root = "logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_6ic_seg20_stitched"
"""

log_path = "pbt_wordcraft_10ic_seg10_log.txt"
log_root = "logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_10ic_seg10"
output_root = "logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_10ic_seg10_stitched"


def parse_pbt_log(log_path):
    replacements = defaultdict(dict)
    current_segment = None
    trial_ids = set()
    segment_start_lines = []

    with open(log_path, "r") as f:
        for line in f:
            # Detect segment start
            if "Starting segment" in line:
                segment_start_lines.append(line)
                seg_start_match = re.search(r"Starting segment (\d+) of (\d+)", line)
                if seg_start_match:
                    current_segment = int(seg_start_match.group(1))

            # Detect replacements
            replace_match = re.search(
                r"Replacing trial (\d+) with a copy of trial (\d+)", line
            )
            if replace_match and current_segment is not None:
                target, source = int(replace_match.group(1)), int(
                    replace_match.group(2)
                )
                replacements[current_segment][target] = source

            # Detect any trial references
            trial_match = re.search(r"Trial (\d+)", line)
            if trial_match:
                trial_ids.add(int(trial_match.group(1)))

    # Get total number of segments from segment starts
    num_segments = 0
    for line in segment_start_lines:
        match = re.search(r"Starting segment (\d+) of (\d+)", line)
        if match:
            seg_num = int(match.group(1))
            num_segments = max(num_segments, seg_num)

    trial_ids = sorted(list(trial_ids))

    # Fill in default (self) replacements for all trials where missing
    for seg in range(1, num_segments + 1):
        for trial in trial_ids:
            if trial not in replacements[seg]:
                replacements[seg][trial] = trial

    return replacements, trial_ids, num_segments


def build_lineages(replacements, trial_ids, num_segments):
    lineages = {trial: [] for trial in trial_ids}
    for trial in trial_ids:
        for seg in range(1, num_segments + 1):
            # Simply: where did trial_i pull its segment `seg` logs from?
            source_trial = replacements[seg][trial]
            lineages[trial].append((seg, source_trial))
    return lineages


def print_lineages(lineages):
    print("\n=== Trial Lineages ===")
    header = ["Trial"] + [
        f"Seg {s}" for s in range(1, len(next(iter(lineages.values()))) + 1)
    ]
    print("\t".join(header))
    for trial, lineage in sorted(lineages.items()):
        lineage_ids = [str(source) for _, source in lineage]
        print(f"{trial}\t" + "\t".join(lineage_ids))


def copy_log_segments_cumulative(lineages, log_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    stitched_dirs = {
        trial: os.path.join(output_root, f"trial_{trial}") for trial in lineages
    }

    # Initialize empty dirs for all trials
    for trial_dir in stitched_dirs.values():
        os.makedirs(trial_dir, exist_ok=True)

    num_segments = len(next(iter(lineages.values())))

    for seg_idx in range(num_segments):
        segment = seg_idx + 1
        print(f"\n=== Segment {segment} ===")

        for trial, lineage in sorted(lineages.items()):
            _, source_trial = lineage[seg_idx]
            dest_dir = stitched_dirs[trial]
            source_dir = stitched_dirs[source_trial]

            # If this trial was replaced (i.e., source ≠ trial), overwrite cumulative logs
            if source_trial != trial:
                print(
                    f"[Replace] Trial {trial} ← Trial {source_trial} at segment {segment}"
                )
                # Remove old cumulative directory and copy from source
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)

            # Now add the current segment log from raw logs
            src_seg = log_root + f"_trial_{trial}" + f"_segment_{segment}"
            dst_seg = dest_dir + f"/segment_{segment}"

            if not os.path.exists(src_seg):
                print(f"[Warning] Missing: {src_seg}")
                continue
            if os.path.exists(dst_seg):
                print(f"[Skip] Already exists: {dst_seg}")
                continue

            shutil.copytree(src_seg, dst_seg)
            print(f"[Copied] {src_seg} → {dst_seg}")

    # Copy all files from segment directory to the trial directory
    # For each trial directory, copy contents of segment directory into trial directory
    for trial_dir in stitched_dirs.values():
        print(f"Flattening {trial_dir}")
        # Flatten nested directories
        for root, dirs, files in os.walk(trial_dir):
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(trial_dir, file)
                print(f"Moving {src_file} to {dst_file}")
                shutil.move(src_file, dst_file)
        # Now delete the segment directories
        for dir in dirs:
            shutil.rmtree(os.path.join(trial_dir, dir))


# Main execution
if __name__ == "__main__":
    replacements, trial_ids, num_segments = parse_pbt_log(log_path)
    lineages = build_lineages(replacements, trial_ids, num_segments)
    print_lineages(lineages)
    copy_log_segments_cumulative(lineages, log_root, output_root)
