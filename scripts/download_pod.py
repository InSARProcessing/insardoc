import re
import glob
import requests
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Any


def download_orbit(safe_files: list, orbit_dir: str) -> None:
    """
    Download precise orbit files for Sentinel-1 SAFE files.

    Args:
        safe_dir: Directory containing SAFE files
        orbit_dir: Directory to save orbit files
    """

    # Create orbit directory if it doesn't exist
    Path(orbit_dir).mkdir(parents=True, exist_ok=True)

    # Get orbit file list from ASF server
    orbit_list = _get_orbit_list()
    if not orbit_list:
        print("Failed to get orbit file list from ASF server")
        return

    print(f"Start processing orbit files.\n")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")
    # For each SAFE file, download the corresponding orbit file
    for safe_file in safe_files:
        print(f"Processing: {Path(safe_file).name}")

        # Extract mission and timestamp from SAFE filename
        mission, start_time, stop_time = _parse_safe_filename(safe_file)

        if not mission or not start_time:
            print("  Could not parse SAFE filename, skipping")
            continue

        # Calculate date range for orbit search (one day before and after)
        date1 = (start_time - timedelta(days=1)).strftime('%Y%m%d')
        date2 = (start_time + timedelta(days=1)).strftime('%Y%m%d')

        # Find matching orbit file
        orbit_file = _find_matching_orbit(orbit_list, mission, date1, date2)

        if not orbit_file:
            print(f"  No orbit file found for {Path(safe_file).name}")
            continue

        # Check if orbit file already exists in orbit directory
        orbit_path = Path(orbit_dir) / orbit_file

        if orbit_path.exists():
            print(f"  Orbit file already exists: {orbit_file}")
            # Create symlink in current directory
            # _create_symlink(orbit_path, Path.cwd() / orbit_file)
            continue

        # Download orbit file
        success = _download_orbit_file_asf(orbit_file, orbit_path)

        if success:
            print(f"  Successfully downloaded: {orbit_file}")
            # Create symlink in current directory
            # _create_symlink(orbit_path, Path.cwd() / orbit_file)
        else:
            print(f"  Failed to download orbit file for: {Path(safe_file).name}")
        print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")


def _get_orbit_list() -> List[str]:
    """
    Get list of orbit files from AWS .
    """
    url_root = "https://s1qc.asf.alaska.edu/aux_poeorb"

    try:
        session = _create_session()
        response = session.get(url_root, timeout=30)
        response.raise_for_status()

        # Extract all .EOF files from the HTML
        orbit_files = re.findall(r'href="([^"]*\.EOF)"', response.text)
        return orbit_files

    except Exception as e:
        print(f"Failed to get orbit list: {e}")
        return []


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


def _download_orbit_file_asf(orbit_file: str, output_path: Path) -> bool:
    """
    Download orbit file from ASF server.
    """
    url_root = "https://s1qc.asf.alaska.edu/aux_poeorb"
    file_url = f"{url_root}/{orbit_file}"

    try:
        session = _create_session()
        response = session.get(file_url, stream=True, timeout=60)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)

                    # Show progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)

        print()  # New line after progress
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        # Clean up partially downloaded file
        if output_path.exists():
            output_path.unlink()
        return False


def _create_symlink(source: Path, link_path: Path) -> None:
    """
    Create a symbolic link to the orbit file.
    """
    try:
        if link_path.exists():
            link_path.unlink()

        # Use absolute path for source to avoid relative path issues
        source_abs = source.resolve()
        link_path.symlink_to(source_abs)
        print(f"  Created symlink: {link_path.name}")

    except Exception as e:
        print(f"  Failed to create symlink: {e}")


def _list_safe_files(safe_dir: str) -> List[str]:
    """List all SAFE files in the directory."""
    safe_files = []

    # Look for .SAFE directories
    safe_dirs = glob.glob(str(Path(safe_dir) / "*.SAFE"))
    safe_files.extend(safe_dirs)

    # Look for .zip files
    zip_files = glob.glob(str(Path(safe_dir) / "*.zip"))
    safe_files.extend(zip_files)

    return safe_files


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


def _create_session() -> requests.Session:
    """
    Create a requests session with retry strategy.
    """
    session = requests.Session()

    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def check_orbit_availability(safe_dir: str, orbit_dir: str) -> Any:
    """
    Check which SAFE files have corresponding orbit files.

    Returns:
        Dictionary with SAFE files as keys and orbit availability as values
    """
    # Get orbit file list from ASF server
    orbit_list = _get_orbit_list()

    safe_files = _list_safe_files(safe_dir)
    availability = {}

    for safe_file in safe_files:
        mission, start_time, stop_time = _parse_safe_filename(safe_file)

        if not mission or not start_time:
            availability[safe_file] = "Parse failed"
            continue

        # Calculate date range for orbit search
        date1 = (start_time - timedelta(days=1)).strftime('%Y%m%d')
        date2 = (start_time + timedelta(days=1)).strftime('%Y%m%d')

        # Find matching orbit file
        orbit_file = _find_matching_orbit(orbit_list, mission, date1, date2)

        if orbit_file:
            orbit_path = Path(orbit_dir) / orbit_file
            if orbit_path.exists():
                availability[safe_file] = "Available"
            else:
                availability[safe_file] = "Missing (exists on server)"
        else:
            availability[safe_file] = "Not found on server"

    return availability


# ----------------------------------------------------------------
if __name__ == "__main__":
    safe_directory = "/mnt/e/InSAR/Stuttgart/RAW/"
    orbit_directory = "/mnt/e/InSAR/Stuttgart/ORB/"

    # Check availability first
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")
    availability = check_orbit_availability(safe_directory, orbit_directory)
    for safe_file, status in availability.items():
        print(f"{Path(safe_file).name}: {status}")
    print("\n")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")
    # Download missing orbit files
    safe_files = _list_safe_files(safe_directory)
    download_orbit(safe_files, orbit_directory)
