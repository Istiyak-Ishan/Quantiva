import sys
import subprocess
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def main():
    print("=" * 60)
    print("  Quantiva Test Suite")
    print("=" * 60)

    # 1. Unit Tests
    print("\n--- Running Unit Tests ---")
    ret1 = subprocess.run([PYTHON, "-m", "pytest", "tests/unit", "-v", "--tb=short"], cwd=BASE).returncode

    # 2. Integration Tests
    print("\n--- Running Integration Tests ---")
    ret2 = subprocess.run([PYTHON, "-m", "pytest", "tests/integration", "-v", "--tb=short"], cwd=BASE).returncode

    # 3. E2E Tests
    print("\n--- Running E2E Test ---")
    ret3 = subprocess.run([PYTHON, "-m", "pytest", "tests/e2e", "-v", "--tb=short"], cwd=BASE).returncode

    print("\n" + "=" * 60)
    if ret1 == 0 and ret2 == 0 and ret3 == 0:
        print("  ALL TESTS PASSED! Quantiva is working cleanly.")
    else:
        print("  SOME TESTS FAILED. Check details above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

