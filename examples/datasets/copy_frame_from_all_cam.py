import os
import shutil
import argparse

def extract_frame_from_cameras(source_dir, target_dir, frame_number):
    """
    Extracts a specific frame from multiple camera directories, copies them
    to a new directory, and renames them based on their source camera folder.

    The script creates a new subdirectory in the target directory named after
    the zero-padded frame number.

    Args:
        source_dir (str): The root directory containing subdirectories for each camera.
        target_dir (str): The directory where the results will be saved.
        frame_number (int): The frame number to extract (e.g., 100).
    """
    # Format the frame number into a 6-digit string with leading zeros (e.g., 100 -> "000100")
    frame_name = str(frame_number).zfill(6)
    frame_filename = f"{frame_name}.jpg"

    # Define the full path for the output directory
    output_path = os.path.join(target_dir, frame_name)

    # Create the output directory; if it exists, do nothing.
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Successfully created or found result directory: {output_path}")
    except OSError as e:
        print(f"Error: Could not create directory {output_path}. Reason: {e}")
        return

    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # Get a list of all items in the source directory that are directories themselves
    try:
        camera_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    if not camera_dirs:
        print(f"Warning: No camera subdirectories found in '{source_dir}'.")
        return

    print(f"Found {len(camera_dirs)} camera directories. Starting extraction for frame {frame_number}...")

    # Loop through each camera directory
    for camera_name in sorted(camera_dirs):
        # Construct the full path to the source image file
        source_image_path = os.path.join(source_dir, camera_name, frame_filename)

        # Check if the specific frame exists in the camera directory
        if os.path.isfile(source_image_path):
            # Construct the destination path, renaming the file to the camera's name
            destination_image_path = os.path.join(output_path, f"{camera_name}.jpg")

            # Copy the file from the source to the destination
            try:
                shutil.copy2(source_image_path, destination_image_path)
                print(f"  - Copied: '{source_image_path}' -> '{destination_image_path}'")
            except IOError as e:
                print(f"  - Error copying file '{source_image_path}'. Reason: {e}")
        else:
            # If the frame doesn't exist, print a warning and skip it
            print(f"  - Warning: Frame '{frame_filename}' not found in camera directory '{camera_name}'. Skipping.")

    print("\nExtraction process completed.")


if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Extract a specific frame from multiple camera directories and rename it.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add arguments for source directory, target directory, and frame number
    parser.add_argument(
        "-s", "--source_dir",
        type=str,
        required=True,
        help="Path to the directory containing all camera subdirectories."
    )
    parser.add_argument(
        "-t", "--target_dir",
        type=str,
        required=True,
        help="Path to the directory where results will be saved."
    )
    parser.add_argument(
        "-f", "--frame_number",
        type=int,
        required=True,
        help="The frame number to extract from each camera (e.g., 100)."
    )
    
    # Example usage message
    parser.epilog = """
Example usage:
  python extract_frame.py \\
    -s /data/shared/elaheh/4D/4D_scenes/elly/jpeg_q90_new_g07_b005 \\
    -t /path/to/my/results \\
    -f 100
"""

    args = parser.parse_args()

    # Call the main function with the provided arguments
    extract_frame_from_cameras(args.source_dir, args.target_dir, args.frame_number)
