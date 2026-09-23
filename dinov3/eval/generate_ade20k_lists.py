import os

# Your dataset root
ROOT_DIR = '/fp/projects01/ec35/data/ADEChallengeData2016'

def generate_list(split_folder_name, output_filename):
    # The actual folder on disk is 'images/training' or 'images/validation'
    image_dir = os.path.join(ROOT_DIR, 'images', split_folder_name)
    output_path = os.path.join(ROOT_DIR, output_filename)
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return

    print(f"Scanning {image_dir}...")
    lines = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # FIX: We now generate 'training/filename.jpg' instead of 'images/training/filename.jpg'
            # The dataloader handles the 'images' folder automatically.
            relative_path = os.path.join(split_folder_name, filename)
            lines.append(relative_path)
            
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Success! Saved {len(lines)} entries to {output_path}")
    # Print the first line to verify the format
    if lines:
        print(f"Sample entry: {lines[0]}")

if __name__ == "__main__":
    generate_list('training', 'ADE20K_object150_train.txt')
    generate_list('validation', 'ADE20K_object150_val.txt')