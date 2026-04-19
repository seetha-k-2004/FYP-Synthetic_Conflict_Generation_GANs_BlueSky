import os
import shutil

source_dir = r'D:\SEETHA004\bluesky\bluesky-master\output'
dest_dir   = r'D:\Users\SEETHA004\OneDrive - Nanyang Technological University\_FYP\learning GAN\rif_logs'

# Create destination folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Find and move all files starting with "RIF_log_"
moved = 0
for file in os.listdir(source_dir):
    if file.startswith("RIF_log_"):
        src  = os.path.join(source_dir, file)
        dst  = os.path.join(dest_dir, file)
        shutil.move(src, dst)
        print(f"Moved: {file}")
        moved += 1

print(f"\nDone! {moved} files moved to:\n{dest_dir}")