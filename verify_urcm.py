""" 
 URCM (Unified μ-Resonance Cognitive Mesh) Verification Test 
 ============================================================ 
 This script verifies if URCM actually works by running real code. 
 
 Run this: python verify_urcm.js 
 """ 
 
import subprocess 
import sys 
import os 
from pathlib import Path 
 
def find_urcm_directory(): 
     """Find URCM project directory""" 
     possible_names = [ 
         'urcm', 
         'URCM', 
         'Unified μ-Resonance Cognitive Mesh', 
         'unified-resonance', 
         'cognitive-mesh' 
     ] 
     
     current = Path.cwd() 
     
     # Check if current directory IS the project directory
     if (current / 'urcm').exists() and (current / 'tests').exists():
         return current

     for name in possible_names: 
         path = current / name 
         if path.exists() and path.is_dir(): 
             # Verify it looks like a project root (has tests)
             if (path / 'tests').exists():
                 return path 
     
     # Check parent directories 
     for name in possible_names: 
         path = current.parent / name 
         if path.exists() and path.is_dir(): 
             if (path / 'tests').exists():
                 return path 
     
     return None 
 
def check_tests_exist(project_dir): 
     """Check if test files exist""" 
     tests_dir = project_dir / 'tests' 
     if not tests_dir.exists(): 
         return [] 
     
     test_files = list(tests_dir.glob('test_*.py')) + list(tests_dir.glob('*_test.py')) 
     return test_files 
 
def run_pytest(project_dir): 
     """Run pytest and capture output""" 
     os.chdir(project_dir) 
     
     commands = [ 
         ['pytest', 'tests/', '-v', '--tb=short', '-p', 'no:ethereum', '-p', 'no:web3', '-p', 'no:pytest_ethereum'], 
         ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short', '-p', 'no:ethereum', '-p', 'no:web3', '-p', 'no:pytest_ethereum'], 
         ['pytest', '-v', '--tb=short', '-p', 'no:ethereum', '-p', 'no:web3', '-p', 'no:pytest_ethereum'] 
     ] 
     
     for cmd in commands: 
         try: 
             result = subprocess.run( 
                 cmd, 
                 capture_output=True, 
                 text=True, 
                 timeout=60 
             ) 
             return result.returncode, result.stdout, result.stderr 
         except FileNotFoundError: 
             continue 
         except subprocess.TimeoutExpired: 
             return -1, "TIMEOUT", "Tests took longer than 60 seconds" 
     
     return -1, "", "Could not run pytest" 
 
def test_main_module(project_dir): 
     """Try to import and run the main module""" 
     sys.path.insert(0, str(project_dir)) 
     
     # Try to import main module 
     try: 
         import urcm 
         return True, "Module imported successfully" 
     except ImportError as e: 
         return False, f"Import failed: {e}" 
 
def main(): 
     print("=" * 60) 
     print("URCM VERIFICATION TEST") 
     print("=" * 60) 
     
     # Step 1: Find project 
     print("\n[1/4] Finding URCM project...") 
     project_dir = find_urcm_directory() 
     
     if not project_dir: 
         print("❌ FAILED: URCM project not found") 
         print("\nSearched for: urcm, URCM, Unified μ-Resonance Cognitive Mesh") 
         print("Current directory:", Path.cwd()) 
         return 
     
     print(f"✅ Found URCM at: {project_dir}") 
     
     # Step 2: Check for tests 
     print("\n[2/4] Checking for test files...") 
     test_files = check_tests_exist(project_dir) 
     
     if not test_files: 
         print("❌ FAILED: No test files found in tests/ directory") 
         return 
     
     print(f"✅ Found {len(test_files)} test files:") 
     for test_file in test_files[:5]:  # Show first 5 
         print(f"   - {test_file.name}") 
     
     # Step 3: Run tests 
     print("\n[3/4] Running pytest...") 
     print("-" * 60) 
     
     returncode, stdout, stderr = run_pytest(project_dir) 
     
     print(stdout) 
     if stderr and "warning" not in stderr.lower(): 
         print("STDERR:", stderr) 
     
     print("-" * 60) 
     
     if returncode == 0: 
         print("✅ ALL TESTS PASSED") 
     elif returncode == -1: 
         print("❌ FAILED: Could not run tests") 
     else: 
         print(f"❌ FAILED: Tests exited with code {returncode}") 
     
     # Step 4: Test module import 
     print("\n[4/4] Testing module import...") 
     success, message = test_main_module(project_dir) 
     
     if success: 
         print(f"✅ {message}") 
     else: 
         print(f"❌ {message}") 
     
     # Summary 
     print("\n" + "=" * 60) 
     print("SUMMARY") 
     print("=" * 60) 
     print(f"Project found: {project_dir.name}") 
     print(f"Test files: {len(test_files)}") 
     print(f"Tests passed: {'YES' if returncode == 0 else 'NO'}") 
     print(f"Module imports: {'YES' if success else 'NO'}") 
     print("=" * 60) 
 
if __name__ == "__main__": 
     main()
