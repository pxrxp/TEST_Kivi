#!/usr/bin/env python3
"""
Verify package compatibility against target Python/buildozer versions before CI build.
Checks PyPI for Python version support without downloading full packages.
"""

import sys
import json
import urllib.request
from packaging import version as pkg_version
from urllib.error import URLError

# Target configuration
TARGET_PYTHON = "3.14"  # kivy/buildozer:latest uses Python 3.14
REQUIREMENTS = {
    "kivy": ">=2.3.0",
    "plyer": None,  # No version constraint
    "numpy": ">=2.0.0",
    "opencv-python": ">=4.8.1.78",
}

class CompatibilityChecker:
    def __init__(self, target_python: str):
        self.target_python = target_python
        self.issues = []
        self.warnings = []

    def fetch_pypi_info(self, package_name: str) -> dict:
        """Fetch package info from PyPI."""
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read())
        except URLError as e:
            self.issues.append(f"❌ Failed to fetch {package_name}: {e}")
            return {}

    def check_package(self, pkg_name: str, version_spec: str = None) -> bool:
        """
        Check if a package supports the target Python version.
        Returns True if compatible, False otherwise.
        """
        print(f"\n🔍 Checking {pkg_name}...", end=" ")
        data = self.fetch_pypi_info(pkg_name)
        
        if not data:
            return False

        try:
            releases = data.get("releases", {})
            info = data.get("info", {})
            
            # Parse version spec (e.g., ">=2.3.0" -> operator=">=", version="2.3.0")
            if version_spec:
                operator, version_str = self._parse_version_spec(version_spec)
                matching_versions = self._find_matching_versions(releases.keys(), operator, version_str)
            else:
                matching_versions = list(releases.keys())

            if not matching_versions:
                self.issues.append(f"❌ {pkg_name}: No versions match spec {version_spec}")
                return False

            # Check the latest matching version for Python 3.14 support
            latest_compatible = self._get_latest_version(matching_versions)
            
            # Get classifiers and requires_python from latest version
            release_data = releases.get(latest_compatible, [])
            pkg_requires_python = info.get("requires_python", "")
            
            supported = self._check_python_support(pkg_requires_python, self.target_python)
            
            if supported is None:
                self.warnings.append(
                    f"⚠️  {pkg_name} ({latest_compatible}): "
                    f"Python support unclear. Requires: {pkg_requires_python or 'unspecified'}"
                )
                return True  # Assume compatible if unspecified
            elif supported:
                print(f"✅ {latest_compatible} (requires_python: {pkg_requires_python or 'any'})")
                return True
            else:
                self.issues.append(
                    f"❌ {pkg_name} ({latest_compatible}): "
                    f"Does NOT support Python {self.target_python}. Requires: {pkg_requires_python}"
                )
                return False

        except Exception as e:
            self.issues.append(f"❌ Error checking {pkg_name}: {e}")
            return False

    def _parse_version_spec(self, spec: str) -> tuple:
        """Parse version specifier like '>=2.0.0' into (operator, version)."""
        for op in [">=", "<=", "==", "!=", ">", "<", "~="]:
            if spec.startswith(op):
                return op, spec[len(op):].strip()
        return ">=", spec

    def _find_matching_versions(self, versions, operator: str, target_version: str) -> list:
        """Filter versions matching the operator constraint."""
        matching = []
        for v in versions:
            try:
                v_parsed = pkg_version.parse(v)
                target_parsed = pkg_version.parse(target_version)
                
                if operator == ">=":
                    match = v_parsed >= target_parsed
                elif operator == ">":
                    match = v_parsed > target_parsed
                elif operator == "<=":
                    match = v_parsed <= target_parsed
                elif operator == "<":
                    match = v_parsed < target_parsed
                elif operator == "==":
                    match = v_parsed == target_parsed
                elif operator == "!=":
                    match = v_parsed != target_parsed
                elif operator == "~=":
                    # ~= allows compatible versions (e.g., 2.x for 2.0)
                    match = v_parsed >= target_parsed and v_parsed.base_version.split('.')[0] == target_parsed.base_version.split('.')[0]
                else:
                    match = False
                
                if match and not str(v_parsed).endswith(('.dev', '.post')):
                    matching.append(v)
            except:
                continue
        
        return sorted(matching, key=pkg_version.parse, reverse=True)

    def _get_latest_version(self, versions: list) -> str:
        """Get the latest version from a list."""
        try:
            return sorted(versions, key=pkg_version.parse, reverse=True)[0]
        except:
            return versions[0] if versions else "unknown"

    def _check_python_support(self, requires_python: str, target_python: str) -> bool:
        """
        Check if requires_python string supports target_python version.
        Returns: True (supported), False (not supported), None (unclear)
        """
        if not requires_python or requires_python == "":
            return None  # Unspecified, assume compatible
        
        requires_python = requires_python.strip()
        target_parsed = pkg_version.parse(target_python)
        
        try:
            # Handle common formats
            for spec in requires_python.split(","):
                spec = spec.strip()
                for op in [">=", "<=", "==", "!=", ">", "<", "~="]:
                    if spec.startswith(op):
                        version_str = spec[len(op):].strip()
                        version_parsed = pkg_version.parse(version_str)
                        
                        if op == ">=":
                            if target_parsed < version_parsed:
                                return False
                        elif op == ">":
                            if target_parsed <= version_parsed:
                                return False
                        elif op == "<":
                            if target_parsed >= version_parsed:
                                return False
                        elif op == "<=":
                            if target_parsed > version_parsed:
                                return False
                        elif op == "==":
                            if target_parsed != version_parsed:
                                return False
                        break
            return True
        except:
            return None

    def run_checks(self) -> int:
        """Run all compatibility checks. Returns exit code."""
        print(f"{'='*60}")
        print(f"🔎 Package Compatibility Checker")
        print(f"Target Python: {self.target_python} (kivy/buildozer:latest)")
        print(f"{'='*60}")

        all_pass = True
        for pkg, spec in REQUIREMENTS.items():
            if not self.check_package(pkg, spec):
                all_pass = False

        print(f"\n{'='*60}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"  {w}")
        
        if self.issues:
            print("\n❌ ISSUES FOUND:")
            for issue in self.issues:
                print(f"  {issue}")
            print(f"\n{'='*60}")
            print("❌ COMPATIBILITY CHECK FAILED")
            print("Fix the issues above before pushing to CI.")
            return 1
        else:
            print("\n✅ ALL CHECKS PASSED")
            print("Safe to push to CI!")
            print(f"{'='*60}")
            return 0

if __name__ == "__main__":
    checker = CompatibilityChecker(TARGET_PYTHON)
    sys.exit(checker.run_checks())
