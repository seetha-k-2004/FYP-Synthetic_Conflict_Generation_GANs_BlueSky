import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from shapely.geometry import LineString, Point
from collections import defaultdict

# Configuration
SCENARIO_FILES_DIR = r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\Conflict Detection\attempt2\2D"
OUTPUT_DIR = r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\Conflict Detection\attempt2\2d_intersections"
POINT_INTERSECTIONS_DIR = r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\Conflict Detection\attempt2\2d_intersections\point_intersections"
ENDPOINT_EXCLUSION_FRACTION = 0.05
EXCLUDED_AIRCRAFT_IDS = {
    "NEP346",
    "AFR256",
    "QFA581",
    "SIA21",
    "SIA212",
    "SIA298",
    "SIA278",
    "SIA256",
    "SIA238",
    "SIA232",
    "PAL502",
    "TGW7",
}

# Create output directories if they don't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(POINT_INTERSECTIONS_DIR).mkdir(parents=True, exist_ok=True)

def parse_scenario_file(file_path):
    """
    Parse a BlueSky scenario file and extract aircraft trajectories.
    
    Returns:
        dict: Dictionary with aircraft callsigns as keys and list of (lat, lon) tuples as values
    """
    trajectories = defaultdict(list)
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse ADDWPT command: "HH:MM:SS.SS> ADDWPT callsign lat lon"
                addwpt_match = re.search(r'ADDWPT\s+(\w+)\s+([\d.-]+)\s+([\d.-]+)', line)
                if addwpt_match:
                    callsign = addwpt_match.group(1)
                    lat = float(addwpt_match.group(2))
                    lon = float(addwpt_match.group(3))
                    trajectories[callsign].append((lat, lon))
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}
    
    return dict(trajectories)

def create_linestrings(trajectories):
    """
    Convert trajectories to Shapely LineString objects.
    
    Args:
        trajectories (dict): Dictionary of callsign -> list of (lat, lon) tuples
    
    Returns:
        dict: Dictionary of callsign -> LineString object
    """
    linestrings = {}
    
    for callsign, points in trajectories.items():
        if len(points) >= 2:
            # Swap to (lon, lat) for Shapely
            coords = [(lon, lat) for lat, lon in points]
            linestrings[callsign] = LineString(coords)
    
    return linestrings

def find_intersections(linestrings):
    """
    Find all pairwise intersections using permutation with lexicographical ordering.
    
    Args:
        linestrings (dict): Dictionary of callsign -> LineString object
    
    Returns:
        list: List of tuples (callsign1, callsign2, intersection_geometry, intersection_type)
    """
    intersections = []
    excluded_near_endpoints = 0
    callsigns = sorted(list(linestrings.keys()))
    
    # Use combinations to ensure each pair is checked only once
    for callsign1, callsign2 in combinations(callsigns, 2):
        line1 = linestrings[callsign1]
        line2 = linestrings[callsign2]
        
        # Check if lines intersect
        if line1.intersects(line2):
            intersection_geom = line1.intersection(line2)
            
            # Classify intersection type
            if isinstance(intersection_geom, Point):
                intersection_type = "Point"
                coords = list(intersection_geom.coords)[0]

                # Exclude point intersections too close to route start/end,
                # which are usually airport-area overlaps.
                line1_length = line1.length
                line2_length = line2.length
                if line1_length > 0 and line2_length > 0:
                    t1 = line1.project(intersection_geom) / line1_length
                    t2 = line2.project(intersection_geom) / line2_length
                    near_endpoint = (
                        t1 <= ENDPOINT_EXCLUSION_FRACTION
                        or t1 >= (1 - ENDPOINT_EXCLUSION_FRACTION)
                        or t2 <= ENDPOINT_EXCLUSION_FRACTION
                        or t2 >= (1 - ENDPOINT_EXCLUSION_FRACTION)
                    )
                    if near_endpoint:
                        excluded_near_endpoints += 1
                        continue
            elif isinstance(intersection_geom, LineString):
                intersection_type = "LineString"
                coords = list(intersection_geom.coords)
            else:
                intersection_type = "Complex"
                coords = str(intersection_geom)
            
            intersections.append({
                'callsign1': callsign1,
                'callsign2': callsign2,
                'intersection_type': intersection_type,
                'intersection_coords': coords,
                'intersection_geometry': intersection_geom
            })

    print(
        f"Excluded {excluded_near_endpoints} point intersections near route start/end "
        f"(threshold={ENDPOINT_EXCLUSION_FRACTION:.0%})."
    )

    return intersections

def save_intersections(intersections, scenario_name):
    """
    Save intersection results to CSV and text files.
    
    Args:
        intersections (list): List of intersection dictionaries
        scenario_name (str): Name of the scenario file
    """
    scenario_basename = scenario_name.replace('.scn', '')
    
    # Save to CSV
    if intersections:
        # Prepare data for CSV
        csv_data = []
        for inter in intersections:
            csv_data.append({
                'Aircraft_1': inter['callsign1'],
                'Aircraft_2': inter['callsign2'],
                'Intersection_Type': inter['intersection_type'],
                'Coordinates': str(inter['intersection_coords'])
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(OUTPUT_DIR, f"{scenario_basename}_intersections.csv")
        df.to_csv(csv_path, index=False)
        print(f"  OK Saved CSV: {scenario_basename}_intersections.csv ({len(intersections)} intersections)")
        
        # Save to text file (summary)
        txt_path = os.path.join(OUTPUT_DIR, f"{scenario_basename}_intersections.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Intersection Summary for {scenario_name}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total intersections found: {len(intersections)}\n\n")
            
            for i, inter in enumerate(intersections, 1):
                f.write(f"Intersection {i}:\n")
                f.write(f"  Aircraft Pair: {inter['callsign1']} - {inter['callsign2']}\n")
                f.write(f"  Type: {inter['intersection_type']}\n")
                f.write(f"  Coordinates: {inter['intersection_coords']}\n\n")
        
        print(f"  OK Saved TXT: {scenario_basename}_intersections.txt")
    else:
        print(f"  o No intersections found in {scenario_name}")

def visualize_intersections(trajectories, linestrings, intersections, scenario_name):
    """
    Visualize flight paths and intersection points.
    
    Args:
        trajectories (dict): Dictionary of callsign -> list of (lat, lon) tuples
        linestrings (dict): Dictionary of callsign -> LineString object
        intersections (list): List of intersection dictionaries
        scenario_name (str): Name of the scenario file
    """
    if not intersections:
        return
    
    plt.figure(figsize=(14, 10))
    
    # Plot each flight path
    colors = plt.cm.tab20(range(len(trajectories)))
    color_map = {callsign: colors[i] for i, callsign in enumerate(sorted(trajectories.keys()))}
    
    for callsign, points in trajectories.items():
        lats, lons = zip(*points)
        plt.plot(lons, lats, marker='o', linestyle='-', label=callsign, 
                color=color_map[callsign], alpha=0.7, markersize=3)
    
    # Highlight intersection points
    for inter in intersections:
        if inter['intersection_type'] == 'Point':
            lon, lat = inter['intersection_coords']
            plt.plot(lon, lat, 'r*', markersize=20, label='Intersection' if inter == intersections[0] else '')
        elif inter['intersection_type'] == 'LineString':
            coords = inter['intersection_coords']
            lons, lats = zip(*coords)
            plt.plot(lons, lats, 'r-', linewidth=3, alpha=0.5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Flight Paths and 2D Intersections\n{scenario_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save visualization
    scenario_basename = scenario_name.replace('.scn', '')
    plot_path = os.path.join(OUTPUT_DIR, f"{scenario_basename}_intersections.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK Saved visualization: {scenario_basename}_intersections.png")

def process_scenario_file(file_path):
    """
    Process a single scenario file for intersections.
    
    Args:
        file_path (str): Path to the scenario file
    """
    scenario_name = os.path.basename(file_path)
    print(f"\nProcessing: {scenario_name}")
    
    # Parse the scenario file
    trajectories = parse_scenario_file(file_path)
    
    if not trajectories:
        print(f"  x No trajectories found")
        return
    
    print(f"  Found {len(trajectories)} aircraft trajectories")
    
    # Convert to LineStrings
    linestrings = create_linestrings(trajectories)
    
    if len(linestrings) < 2:
        print(f"  x Need at least 2 aircraft to find intersections")
        return
    
    # Find intersections
    intersections = find_intersections(linestrings)
    
    # Save results
    save_intersections(intersections, scenario_name)
    
    # Visualize (only if intersections found)
    if intersections:
        try:
            visualize_intersections(trajectories, linestrings, intersections, scenario_name)
        except Exception as e:
            print(f"  ! Could not visualize: {e}")

def create_point_intersection_scenarios(intersections, file_aircraft_map):
    """
    Create BlueSky scenario files for point intersections.
    Each file loads the two aircraft that intersect.
    
    Args:
        intersections (list): List of intersection dictionaries
        file_aircraft_map (dict): Map of aircraft callsign to scenario file name
    """
    point_intersections = [
        inter
        for inter in intersections
        if inter['intersection_type'] == 'Point'
        and inter['callsign1'] not in EXCLUDED_AIRCRAFT_IDS
        and inter['callsign2'] not in EXCLUDED_AIRCRAFT_IDS
    ]
    
    if not point_intersections:
        print("No point intersections to save")
        return
    
    print(f"\nCreating {len(point_intersections)} point intersection scenario files...")
    
    for i, inter in enumerate(point_intersections, 1):
        callsign1 = inter['callsign1']
        callsign2 = inter['callsign2']
        file1 = file_aircraft_map.get(callsign1, '')
        file2 = file_aircraft_map.get(callsign2, '')
        
        if not file1 or not file2:
            continue
        
        # Create scenario file name
        scenario_name = f"{i:03d}_{callsign1}_{callsign2}.scn"
        scenario_path = os.path.join(POINT_INTERSECTIONS_DIR, scenario_name)
        
        # Get intersection coordinates
        if inter['intersection_type'] == 'Point':
            lon, lat = inter['intersection_coords']
            coords_str = f"{lat}, {lon}"
        
        # Create scenario file content
        content = f"""# Point Intersection Scenario
# Aircraft 1: {callsign1} (from {file1})
# Aircraft 2: {callsign2} (from {file2})
# Intersection Point: Lat={lat:.6f}, Lon={lon:.6f}

00:00:00.00>TRAIL ON
00:00:00.00>DT 1.0
00:00:00.00>PLUGIN RIFLOG_THUNDER
00:00:00.00>ASAS ON

# ---- Your two flights ----
00:00:00.00>CALL {file1}
00:00:00.00>CALL {file2}
"""
        
        # Write scenario file
        try:
            with open(scenario_path, 'w') as f:
                f.write(content)
            print(f"  OK Created: {scenario_name}")
        except Exception as e:
            print(f"  x Error creating {scenario_name}: {e}")
    
    print(f"OK Point intersection scenarios saved to: {POINT_INTERSECTIONS_DIR}")

def main():
    """
    Main function to process all scenario files together.
    Since each file contains one aircraft, we need to combine them all.
    """
    print("=" * 70)
    print("2D INTERSECTION DETECTION")
    print("=" * 70)
    print(f"\nScenario Files Directory: {SCENARIO_FILES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    # Find all scenario files
    scenario_files = []
    for file in os.listdir(SCENARIO_FILES_DIR):
        if file.endswith('.scn'):
            scenario_files.append(os.path.join(SCENARIO_FILES_DIR, file))
    
    scenario_files.sort()
    print(f"Found {len(scenario_files)} scenario files")
    print(f"Processing all files together to find 2D intersections...\n")
    
    # Parse all scenario files and combine trajectories
    all_trajectories = {}
    file_aircraft_map = {}  # Map aircraft to file names
    excluded_trajectories = 0
    
    print("Parsing scenario files...")
    for i, file_path in enumerate(scenario_files, 1):
        try:
            trajectories = parse_scenario_file(file_path)
            for callsign, points in trajectories.items():
                if callsign in EXCLUDED_AIRCRAFT_IDS:
                    excluded_trajectories += 1
                    continue
                if len(points) >= 2:
                    all_trajectories[callsign] = points
                    file_aircraft_map[callsign] = os.path.basename(file_path)
        except Exception as e:
            pass
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(scenario_files)} files... ({len(all_trajectories)} aircraft found)")
    
    print(f"\nTotal aircraft trajectories found: {len(all_trajectories)}")
    print(f"Excluded trajectories by aircraft ID filter: {excluded_trajectories}\n")
    
    if len(all_trajectories) < 2:
        print("Error: Need at least 2 aircraft to find intersections")
        return
    
    # Create linestrings
    print("Creating LineString objects...")
    linestrings = create_linestrings(all_trajectories)
    print(f"Valid LineStrings created: {len(linestrings)}\n")
    
    # Find intersections
    print("Finding 2D intersections (using lexicographical ordering)...")
    intersections = find_intersections(linestrings)
    
    if not intersections:
        print("No 2D intersections found in the dataset.")
        return
    
    print(f"Found {len(intersections)} intersection(s)\n")
    
    # Save results for all intersections
    print("Saving results...")
    if intersections:
        # Prepare data for CSV
        csv_data = []
        for inter in intersections:
            csv_data.append({
                'Aircraft_1': inter['callsign1'],
                'Aircraft_2': inter['callsign2'],
                'File_Aircraft_1': file_aircraft_map.get(inter['callsign1'], 'Unknown'),
                'File_Aircraft_2': file_aircraft_map.get(inter['callsign2'], 'Unknown'),
                'Intersection_Type': inter['intersection_type'],
                'Coordinates': str(inter['intersection_coords'])
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(OUTPUT_DIR, "all_intersections.csv")
        df.to_csv(csv_path, index=False)
        print(f"OK Saved comprehensive CSV: all_intersections.csv ({len(intersections)} intersections)")
        
        # Save to text file (summary)
        txt_path = os.path.join(OUTPUT_DIR, "all_intersections_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("2D INTERSECTION DETECTION - COMPLETE DATASET\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total files processed: {len(scenario_files)}\n")
            f.write(f"Total aircraft trajectories: {len(all_trajectories)}\n")
            f.write(f"Total intersections found: {len(intersections)}\n\n")
            
            f.write("INTERSECTION DETAILS:\n")
            f.write("-" * 70 + "\n\n")
            
            for i, inter in enumerate(intersections, 1):
                f.write(f"Intersection {i}:\n")
                f.write(f"  Aircraft Pair: {inter['callsign1']} - {inter['callsign2']}\n")
                f.write(f"  Type: {inter['intersection_type']}\n")
                f.write(f"  Coordinates: {inter['intersection_coords']}\n")
                f.write(f"  Source Files:\n")
                f.write(f"    {inter['callsign1']}: {file_aircraft_map.get(inter['callsign1'], 'Unknown')}\n")
                f.write(f"    {inter['callsign2']}: {file_aircraft_map.get(inter['callsign2'], 'Unknown')}\n\n")
        
        print(f"OK Saved summary TXT: all_intersections_summary.txt")
        
        # Visualize all intersections
        try:
            print("OK Generating visualization...")
            visualize_intersections(all_trajectories, linestrings, intersections, "Complete_Dataset_All_Intersections")
            print(f"OK Saved visualization: Complete_Dataset_All_Intersections.png")
        except Exception as e:
            print(f"! Could not visualize: {e}")
        
        # Create scenario files for point intersections
        create_point_intersection_scenarios(intersections, file_aircraft_map)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files processed: {len(scenario_files)}")
    print(f"Total aircraft trajectories: {len(all_trajectories)}")
    print(f"Total 2D intersections found: {len(intersections)}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
