import os
import subprocess

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False)

print("--- 1. Removing Boilerplate ---")
boilers = ['CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'MODEL_CARD.md', 'DATASETS.md', 'hubconf.py']
for b in boilers:
    if os.path.exists(b):
        run(f"git rm -f {b}")

print("\n--- 2. Renaming Core Package ---")
if os.path.exists('dinov3'):
    run("git mv dinov3 cam")

print("\n--- 3. Removing Bloat (Eval, Hub, Thirdparty) ---")
heavy_dirs = [
    'cam/eval/detection',
    'cam/eval/segmentation',
    'cam/eval/text',
    'cam/thirdparty',
    'cam/hub'
]
for d in heavy_dirs:
    if os.path.exists(d):
        run(f"git rm -rf {d}")

print("\n--- 4. Renaming Core Files ---")
renames = {
    'cam/train/cam_meta_arch.py': 'cam/train/cam_meta_arch.py',
    'cam/layers/cam_head.py': 'cam/layers/cam_head.py',
    'cam/loss/cam_loss.py': 'cam/loss/cam_loss.py',
}
for src, dst in renames.items():
    if os.path.exists(src):
        run(f"git mv {src} {dst}")

print("\n--- 5. Updating Imports and References ---")
replacements = {
    'from cam': 'from cam',
    'import cam': 'import cam',
    'cam.': 'cam.',
    'cam/': 'cam/',
    'cam_meta_arch': 'cam_meta_arch',
    'CAMMetaArch': 'CAMMetaArch',
    'cam_head': 'cam_head',
    'CAMHead': 'CAMHead',
    'cam_loss': 'cam_loss',
    'CAMLoss': 'CAMLoss'
}

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    orig_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    if content != orig_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")
        run(f"git add {filepath}")

for root, dirs, files in os.walk('.'):
    if '.git' in root or 'experiments' in root or 'logs' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith(('.py', '.yaml', '.sh', '.md', '.toml', '.txt')):
            filepath = os.path.join(root, file)
            try:
                replace_in_file(filepath)
            except Exception as e:
                pass

print("\n--- Done! ---")
