
#!/usr/bin/env python3
"""
Cleanup Unused Files Script
===========================

Identifies and optionally removes unused files from the ValiCred-AI codebase.
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Set, Dict

class UnusedFileDetector:
    """Detects unused files in the codebase"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.used_files = set()
        self.all_python_files = set()
        self.import_graph = {}
        
    def scan_imports(self) -> Dict[str, List[str]]:
        """Scan all Python files for imports"""
        import_map = {}
        
        for py_file in self.root_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            self.all_python_files.add(str(py_file.relative_to(self.root_dir)))
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                import_map[str(py_file.relative_to(self.root_dir))] = imports
                
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
        
        return import_map
    
    def find_unused_files(self) -> List[str]:
        """Find unused Python files"""
        import_map = self.scan_imports()
        
        # Mark main entry points as used
        entry_points = ["app.py"]
        used_files = set(entry_points)
        
        # Recursively find all imported files
        to_process = list(entry_points)
        
        while to_process:
            current_file = to_process.pop(0)
            
            if current_file not in import_map:
                continue
            
            for import_name in import_map[current_file]:
                # Convert import to file path
                potential_files = self._import_to_files(import_name)
                
                for file_path in potential_files:
                    if file_path in self.all_python_files and file_path not in used_files:
                        used_files.add(file_path)
                        to_process.append(file_path)
        
        # Find unused files
        unused_files = self.all_python_files - used_files
        return sorted(list(unused_files))
    
    def _import_to_files(self, import_name: str) -> List[str]:
        """Convert import name to possible file paths"""
        possible_files = []
        
        # Handle relative imports
        if import_name.startswith('.'):
            return possible_files
        
        # Convert module path to file path
        parts = import_name.split('.')
        
        # Try as direct file
        file_path = '/'.join(parts) + '.py'
        possible_files.append(file_path)
        
        # Try as package
        package_path = '/'.join(parts) + '/__init__.py'
        possible_files.append(package_path)
        
        # Try in src directory
        src_file_path = 'src/' + '/'.join(parts) + '.py'
        possible_files.append(src_file_path)
        
        src_package_path = 'src/' + '/'.join(parts) + '/__init__.py'
        possible_files.append(src_package_path)
        
        return possible_files
    
    def find_unused_directories(self) -> List[str]:
        """Find directories that contain only unused files"""
        unused_files = self.find_unused_files()
        unused_dirs = []
        
        # Group files by directory
        dir_files = {}
        for file_path in self.all_python_files:
            dir_path = str(Path(file_path).parent)
            if dir_path not in dir_files:
                dir_files[dir_path] = []
            dir_files[dir_path].append(file_path)
        
        # Check if all files in directory are unused
        for dir_path, files in dir_files.items():
            if dir_path == '.' or dir_path == 'src':
                continue
                
            all_unused = all(f in unused_files for f in files)
            if all_unused and len(files) > 0:
                unused_dirs.append(dir_path)
        
        return sorted(unused_dirs)

def main():
    """Main execution function"""
    detector = UnusedFileDetector()
    
    print("🔍 Scanning for unused files in ValiCred-AI codebase...")
    print("=" * 60)
    
    # Find unused Python files
    unused_files = detector.find_unused_files()
    
    print(f"\n📁 Found {len(unused_files)} potentially unused Python files:")
    print("-" * 40)
    
    for file_path in unused_files:
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        print(f"  • {file_path} ({file_size} bytes)")
    
    # Find unused directories
    unused_dirs = detector.find_unused_directories()
    
    print(f"\n📂 Found {len(unused_dirs)} potentially unused directories:")
    print("-" * 40)
    
    for dir_path in unused_dirs:
        print(f"  • {dir_path}/")
    
    # Check for other unused files
    print(f"\n📄 Other potentially unused files:")
    print("-" * 40)
    
    other_unused = [
        "docs/ONBOARDING.md",
        "src/backend/fastapi_server.py", 
        "src/data/real_data_loader.py",
        "src/ui/configuration_panel.py",
        "src/utils/workflow_manager.py"
    ]
    
    for file_path in other_unused:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            print(f"  • {file_path} ({file_size} bytes)")
    
    print(f"\n💡 Recommendations:")
    print("-" * 40)
    print("  • Review unused files before deletion")
    print("  • Some files may be used by configuration or templates")
    print("  • Consider moving useful components to active modules")
    print("  • Update documentation to reflect removed components")
    
    # Interactive cleanup option
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print(f"\n❓ Would you like to interactively remove unused files? (y/N): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            for file_path in unused_files:
                print(f"\n🗑️  Remove {file_path}? (y/N): ", end='')
                if input().strip().lower() == 'y':
                    try:
                        Path(file_path).unlink()
                        print(f"   ✅ Removed {file_path}")
                    except Exception as e:
                        print(f"   ❌ Error removing {file_path}: {e}")

if __name__ == "__main__":
    main()
