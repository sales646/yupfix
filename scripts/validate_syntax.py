"""Validate all Python files compile without errors."""
import py_compile
import sys
from pathlib import Path

def validate_all_python_files(root_dir: str = "."):
    """Check all .py files for syntax errors."""
    root = Path(root_dir)
    errors = []
    checked = 0
    
    # Find all Python files (exclude venv)
    for py_file in root.rglob("*.py"):
        if ".venv" in str(py_file) or "venv" in str(py_file):
            continue
        if "__pycache__" in str(py_file):
            continue
            
        checked += 1
        try:
            py_compile.compile(str(py_file), doraise=True)
            print(f"  ✅ {py_file}")
        except py_compile.PyCompileError as e:
            errors.append((py_file, str(e)))
            print(f"  ❌ {py_file}")
            print(f"     {e}")
    
    print(f"\n{'='*50}")
    print(f"Checked: {checked} files")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\n❌ SYNTAX ERRORS FOUND:")
        for file, error in errors:
            print(f"  {file}: {error}")
        return False
    else:
        print("\n✅ ALL FILES VALID")
        return True


if __name__ == "__main__":
    success = validate_all_python_files(".")
    sys.exit(0 if success else 1)
