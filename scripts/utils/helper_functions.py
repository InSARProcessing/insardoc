from pathlib import Path
from datetime import datetime
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
