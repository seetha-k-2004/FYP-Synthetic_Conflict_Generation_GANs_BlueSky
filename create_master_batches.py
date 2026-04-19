import os
from pathlib import Path


def create_master_batches(input_dir, output_base_dir, batch_size=20):
    """
    Create master scenario files with a fixed number of .scn files per batch.

    Args:
        input_dir: Directory containing all .scn files
        output_base_dir: Directory where master batch files will be created
        batch_size: Number of aircraft (.scn CALL lines) per master file
    """
    
    # Get all .scn files and sort them
    scn_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.scn')])
    
    if not scn_files:
        print(f"No .scn files found in {input_dir}")
        return
    
    print(f"Found {len(scn_files)} .scn files")
    
    # Create output base directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Template for master scenario file
    master_template = """00:00:00.00>TRAIL ON
00:00:00.00>DT 10
00:00:00.00>PLUGIN TRACKLOG
00:00:00.00>ASAS ON
"""
    
    num_batches = (len(scn_files) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(scn_files))
        batch_files = scn_files[start_idx:end_idx]

        master_content = master_template
        for scn_file in batch_files:
            master_content += f"00:00:00.00>CALL {scn_file}\n"

        master_file = Path(output_base_dir) / f"master_batch{batch_num + 1}.scn"
        with open(master_file, 'w') as f:
            f.write(master_content)

        print(f"✓ Created batch {batch_num + 1}: {len(batch_files)} files -> {master_file.name}")

    print(f"\nMaster files created successfully in: {output_base_dir}")
    print(f"Total batches: {num_batches}")


if __name__ == "__main__":
    input_dir = r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\Conflict Detection\attempt2\2D"
    output_base_dir = r"d:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\Conflict Detection\attempt2\2D_batches"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_base_dir}\n")
    
    create_master_batches(input_dir, output_base_dir, batch_size=20)
