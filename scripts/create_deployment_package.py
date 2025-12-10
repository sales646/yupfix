"""Create a clean deployment package for the server."""
import shutil
from pathlib import Path
import zipfile

def create_package():
    """Create deployment zip without venv/data/models."""
    
    print("Creating deployment package...")
    
    # Files/folders to include
    include = [
        'src/',
        'config/',
        'scripts/',
        'tests/',
        'requirements.txt',
        'README.md',
        '.gitignore',
    ]
    
    # Create temp directory
    deploy_dir = Path('yup250_deploy')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy files
    for item in include:
        src = Path(item)
        if src.exists():
            dst = deploy_dir / item
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '*.pyo', '.git'
                ))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            print(f"  ✅ {item}")
        else:
            print(f"  ⚠️ {item} not found")
    
    # Create data directory structure
    (deploy_dir / 'data' / 'train').mkdir(parents=True)
    (deploy_dir / 'data' / 'val').mkdir(parents=True)
    (deploy_dir / 'models' / 'checkpoints').mkdir(parents=True)
    (deploy_dir / 'logs' / 'training').mkdir(parents=True)
    
    # Create zip
    zip_path = 'yup250_deploy.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in deploy_dir.rglob('*'):
            if file.is_file():
                zf.write(file, file.relative_to(deploy_dir))
    
    # Cleanup
    shutil.rmtree(deploy_dir)
    
    # Get size
    size_mb = Path(zip_path).stat().st_size / 1024 / 1024
    
    print(f"\n✅ Created: {zip_path} ({size_mb:.1f} MB)")
    print("\nUpload this to your A100 server!")


if __name__ == "__main__":
    create_package()
