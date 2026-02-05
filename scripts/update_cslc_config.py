import os
import sys
import glob
import argparse
import yaml
from datetime import datetime, timedelta
from .utils import _parse_safe_filename, _find_matching_orbit


def main():
    parser = argparse.ArgumentParser(description="Update S1 CSLC configuration file (yaml)")

    # Required argument: configuration file path
    parser.add_argument("config_file", help="Path to the .yaml configuration file to modify")

    # Optional arguments: define according to your yaml structure
    parser.add_argument("--safe-dir", type=str, required=True, help="SAFE file directory")
    parser.add_argument("--orbit-dir", type=str, required=True, help="Orbit (EOF) file directory")
    parser.add_argument("--dem-file", help="DEM file path")
    parser.add_argument("--product-path", help="Product output directory")
    parser.add_argument("--scratch-path", help="Temporary file directory")
    parser.add_argument("--sas-file", type=str, help="SAS file path")
    parser.add_argument("--gpu-enabled", type=str, choices=['True', 'False'], help="Whether to enable GPU")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file not found {args.config_file}")
        sys.exit(1)

    # Read YAML
    print(f"Reading: {args.config_file} ...")
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get reference to groups node for easier access later
    try:
        groups = config['runconfig']['groups']
    except KeyError:
        print("Error: Configuration file structure is not as expected (missing runconfig.groups)")
        sys.exit(1)

    # --- Update input_file_group ---
    safe_files = glob.glob(os.path.join(args.safe_dir, "S1*_IW_*"))
    print(f"Updating safe_file_path: {len(safe_files)} files")
    groups['input_file_group']['safe_file_path'] = safe_files

    orbit_files = glob.glob(os.path.join(args.orbit_dir, "*.EOF"))
    orbit_list = []
    for sfile in safe_files:
        find_orbit = 0
        mission, start_time, stop_time = _parse_safe_filename(sfile)
        # Extract mission short name (A or B)
        mission_short = mission[-1]
        # Calculate date range for orbit search (one day before and after)
        date1 = (start_time - timedelta(days=1)).strftime('%Y%m%d')
        date2 = (start_time + timedelta(days=1)).strftime('%Y%m%d')
        # Search for matching orbit file
        for candidate in orbit_files:
            # Check if file matches mission and contains both dates
            if (f"S1{mission_short}" in candidate and
                date1 in candidate and
                    date2 in candidate):
                find_orbit = 1
                orbit_list.append(candidate)
        if not find_orbit:
            print(f"[Warning]: orbit file is missing for file: {sfile}")
        print(f"Updating orbit_file_path: {len(orbit_list)} files")
        groups['input_file_group']['orbit_file_path'] = orbit_list

    # --- Update dynamic_ancillary_file_group ---
    if args.dem_file:
        print(f"Updating dem_file: {args.dem_file}")
        groups['dynamic_ancillary_file_group']['dem_file'] = args.dem_file

    # --- Update product_path_group ---
    if args.product_path:
        os.makedirs(args.product_path, exist_ok=True)
        print(f"Updating product_path: {args.product_path}")
        groups['product_path_group']['product_path'] = args.product_path

    if args.scratch_path:
        os.makedirs(args.scratch_path, exist_ok=True)
        print(f"Updating scratch_path: {args.scratch_path}")
        groups['product_path_group']['scratch_path'] = args.scratch_path
    if args.sas_file:
        print(f"Updating sas_output_file: {args.sas_file}")
        groups['product_path_group']['sas_output_file'] = args.sas_file

    # --- Update worker ---
    if args.gpu_enabled:
        # yaml usually expects boolean values, so convert accordingly
        bool_val = True if args.gpu_enabled.lower() == 'true' else False
        print(f"Updating gpu_enabled: {bool_val}")
        groups['worker']['gpu_enabled'] = bool_val

    # Write back to file
    print(f"Saving updates to: {args.config_file}")
    with open(args.config_file, 'w') as f:
        yaml.dump(config, f)

    print("Done!")


if __name__ == "__main__":
    main()
