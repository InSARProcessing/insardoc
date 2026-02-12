import os
import sys
import glob
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, List


def _parse_safe_filename(safe_file: str) -> Tuple[Optional[str], Optional[datetime], Optional[datetime]]:
    """
    Parse SAFE filename to extract mission and timestamps.

    Expected format: S1A_IW_SLC__1SDV_20220101T120000_20220101T120030_041200_04E234_ABCD.SAFE
    or similar patterns.
    """
    basename = Path(safe_file).name

    # Remove .SAFE or .zip extension
    if basename.endswith('.SAFE'):
        basename = basename[:-5]
    elif basename.endswith('.zip'):
        basename = basename[:-4]

    # Split by underscores
    parts = basename.split('_')

    if len(parts) < 6:
        return None, None, None

    # Mission identifier (S1A or S1B)
    mission = parts[0]

    # Timestamps are typically at positions 5 and 6
    # Format: YYYYMMDDTHHMMSS
    start_time_str = parts[5]
    stop_time_str = parts[6]

    # Parse timestamps
    try:
        start_time = datetime.strptime(start_time_str, '%Y%m%dT%H%M%S')
        stop_time = datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S')
        return mission, start_time, stop_time
    except ValueError:
        print(f"  Could not parse timestamps from: {basename}")
        return None, None, None


def _find_matching_orbit(orbit_list: List[str], mission: str, date1: str, date2: str) -> Optional[str]:
    """
    Find orbit file that matches the mission and date range.
    """
    # Extract mission short name (A or B)
    mission_short = mission[-1]

    # Search for matching orbit file
    for orbit_file in orbit_list:
        # Check if file matches mission and contains both dates
        if (f"S1{mission_short}" in orbit_file and
            date1 in orbit_file and
                date2 in orbit_file):
            return orbit_file

    return None


def args_parser():
    parser = argparse.ArgumentParser(description="Update S1 CSLC configuration file (yaml)")
    # Required argument: configuration file path
    parser.add_argument("config_file", dest='config_file', help="Path to the .yaml configuration file to modify")

    # Optional arguments: define according to your yaml structure
    parser.add_argument("--safe-dir", dest='safe_dir', type=str, required=True, help="SAFE file directory")
    parser.add_argument("--orbit-dir", dest='orbit_dir', type=str, required=True, help="Orbit (EOF) file directory")
    parser.add_argument("--dem-file", dest='dem_file', help="DEM file path")
    parser.add_argument("--product-path", dest='product_path', help="Product output directory")
    parser.add_argument("--scratch-path", dest='scratch_path', help="Temporary file directory")
    parser.add_argument("--sas-file", dest='sas_file', type=str, help="SAS file path")
    parser.add_argument("--gpu-enabled", dest='gpu_enabled', action="store_true", help="Whether to enable GPU")
    args = parser.parse_args()

    return args


def main(config_file: str,
         safe_dir: str,
         orbit_dir: str,
         dem_file: str,
         product_path: str,
         scratch_path: str,
         sas_file: str,
         gpu_enabled: bool = True):
    if not os.path.exists(config_file):
        print(
            f"Warning: Configuration file not found {config_file}, Copying default_config.yaml to {config_file}")
        default_config = os.path.join(os.path.dirname(__file__), "default", "s1_cslc_geo.yaml")
        shutil.copy(default_config, config_file)

    # Read YAML
    print(f"Reading: {config_file} ...")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get reference to groups node for easier access later
    try:
        groups = config['runconfig']['groups']
    except KeyError:
        print("Error: Configuration file structure is not as expected (missing runconfig.groups)")
        sys.exit(1)

    # --- Update input_file_group ---
    safe_files = glob.glob(os.path.join(safe_dir, "S1*_IW_*"))
    print(f"Updating safe_file_path: {len(safe_files)} files")
    groups['input_file_group']['safe_file_path'] = safe_files

    orbit_files = glob.glob(os.path.join(orbit_dir, "*.EOF"))
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
    if dem_file:
        print(f"Updating dem_file: {dem_file}")
        groups['dynamic_ancillary_file_group']['dem_file'] = dem_file

    # --- Update product_path_group ---
    if product_path:
        os.makedirs(product_path, exist_ok=True)
        print(f"Updating product_path: {product_path}")
        groups['product_path_group']['product_path'] = product_path

    if scratch_path:
        os.makedirs(scratch_path, exist_ok=True)
        print(f"Updating scratch_path: {scratch_path}")
        groups['product_path_group']['scratch_path'] = scratch_path
    if sas_file:
        print(f"Updating sas_output_file: {sas_file}")
        groups['product_path_group']['sas_output_file'] = sas_file

    # --- Update worker ---
    if gpu_enabled:
        # yaml usually expects boolean values, so convert accordingly
        bool_val = True if gpu_enabled else False
        print(f"Updating gpu_enabled: {bool_val}")
        groups['worker']['gpu_enabled'] = bool_val

    # Write back to file
    print(f"Saving updates to: {config_file}")
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    print("Done!")


if __name__ == "__main__":
    args = args_parser()
    main(args.config_file,
         args.safe_dir,
         args.orbit_dir,
         args.dem_file,
         args.product_path,
         args.scratch_path,
         args.sas_file,
         args.gpu_enabled)
