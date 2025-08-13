#!/usr/bin/env python3
"""
Quick pre-flight check before running the full pipeline
No heavy data loading - just verify setup
"""

import os
import sys
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("TENNIS PIPELINE PRE-FLIGHT CHECK")
    print("="*60)
    
    issues = []
    warnings = []
    
    # 1. Check critical directories
    print("\n📁 Checking directories...")
    dirs_to_check = {
        'Jeff Data': os.path.expanduser('~/Desktop/data/Jeff 6.14.25'),
        'Cache': os.path.expanduser('~/Desktop/data/cache'),
        'Logs': 'logs'
    }
    
    for name, path in dirs_to_check.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            if name == 'Logs':
                os.makedirs(path, exist_ok=True)
                print(f"  ✓ {name}: Created {path}")
            else:
                print(f"  ✗ {name}: NOT FOUND at {path}")
                issues.append(f"Missing {name} directory")
    
    # 2. Check critical files
    print("\n📄 Checking critical files...")
    files_to_check = {
        'Pipeline': 'tennis_updated.py',
        'Runner': 'tennis_pipeline_runner.py',
        'Model': 'model.py',
        'Settings': 'settings.py'
    }
    
    for name, file in files_to_check.items():
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"  ✓ {name}: {file} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {name}: {file} NOT FOUND")
            issues.append(f"Missing {file}")
    
    # 3. Check Jeff's point data files
    print("\n🎾 Checking Jeff's point sequence files...")
    jeff_base = os.path.expanduser('~/Desktop/data/Jeff 6.14.25')
    point_files = [
        'men/charting-m-points-2020s.csv',
        'men/charting-m-points-2010s.csv',
        'men/charting-m-points-to-2009.csv',
        'women/charting-w-points-2020s.csv'
    ]
    
    total_size = 0
    for file in point_files:
        path = os.path.join(jeff_base, file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            total_size += size_mb
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️ {file} not found")
            warnings.append(f"Missing {file}")
    
    print(f"\n  Total point data: {total_size:.1f} MB")
    
    # 4. Check Python modules
    print("\n📦 Checking Python modules...")
    required = ['pandas', 'numpy', 'polars', 'httpx', 'requests_cache', 'bs4']
    missing = []
    
    for module in required:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module}")
            missing.append(module)
    
    if missing:
        issues.append(f"Missing Python modules: {', '.join(missing)}")
        print(f"\n  Install with: pip install {' '.join(missing)}")
    
    # 5. Check API key
    print("\n🔑 Checking API configuration...")
    if os.environ.get('API_TENNIS_KEY'):
        print("  ✓ API_TENNIS_KEY set in environment")
    else:
        # Check if it's hardcoded in the file
        with open('tennis_updated.py', 'r') as f:
            if 'API_TENNIS_KEY' in f.read():
                print("  ✓ API key found in tennis_updated.py")
            else:
                print("  ⚠️ API_TENNIS_KEY not found")
                warnings.append("API key not configured")
    
    # 6. Check point parsing implementation
    print("\n🔍 Checking Jeff point parsing...")
    with open('tennis_updated.py', 'r') as f:
        content = f.read()
        if 'parse_point' in content and 'def parse_point' in content:
            print("  ✓ parse_point function found")
        else:
            print("  ⚠️ No parse_point function - Jeff sequences won't be utilized!")
            warnings.append("Jeff point sequences not being parsed (20% improvement lost)")
    
    # 7. Estimate runtime
    print("\n⏱️ Runtime estimate...")
    print("  Based on data size: 100-150 hours")
    print("  Recommendation: Use nohup or screen")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if issues:
        print("\n❌ CRITICAL ISSUES (must fix):")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️ WARNINGS (should review):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues:
        print("\n✅ READY TO RUN!")
        print("\nCommands:")
        print("  Test run (50 matches): python tennis_updated.py --test")
        print("  Full run: python tennis_pipeline_runner.py")
        print("  Background: nohup python tennis_pipeline_runner.py > pipeline.log 2>&1 &")
        print("  Monitor: python tennis_pipeline_runner.py --monitor")
        return 0
    else:
        print("\n❌ Fix critical issues before running")
        return 1

if __name__ == "__main__":
    sys.exit(main())