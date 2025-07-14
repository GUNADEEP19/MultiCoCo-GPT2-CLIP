#!/usr/bin/env python3
"""
Version compatibility checker for COCONUT LLaVA training.
This script checks your current package versions and provides compatibility information.
"""

import sys
import importlib

def check_package_version(package_name, required_version=None, min_version=None, max_version=None):
    """Check if a package is installed and optionally verify version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        
        if required_version and version != required_version:
            print(f"   ⚠️  Expected: {required_version}")
            return False
        elif min_version and version < min_version:
            print(f"   ⚠️  Should be >= {min_version}")
            return False
        elif max_version and version > max_version:
            print(f"   ⚠️  Should be <= {max_version}")
            return False
        
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def main():
    print("🔍 COCONUT LLaVA Version Compatibility Checker")
    print("=" * 50)
    
    # Your target versions
    target_versions = {
        'torch': '2.1.2+cu121',
        'transformers': '4.37.2',
        'scikit-learn': '1.2.2',
        'matplotlib': '3.10.3',
        'PIL': '10.4.0',  # Pillow
        'yaml': '6.0.2',  # PyYAML
        'wandb': '0.21.0'
    }
    
    print("📦 Checking your current package versions:")
    print()
    
    all_compatible = True
    
    # Check each package
    for package, target_version in target_versions.items():
        if package == 'PIL':
            # Special case for Pillow
            compatible = check_package_version('PIL', target_version)
        elif package == 'yaml':
            # Special case for PyYAML
            compatible = check_package_version('yaml', target_version)
        else:
            compatible = check_package_version(package, target_version)
        
        if not compatible:
            all_compatible = False
    
    print("\n" + "=" * 50)
    
    if all_compatible:
        print("🎉 All packages are compatible with your target versions!")
        print("✅ You can proceed with training.")
    else:
        print("⚠️  Some packages have version mismatches.")
        print("📝 Consider updating to match the target versions.")
    
    print("\n🔧 Known Issues with transformers 4.37.2:")
    print("   - LlavaProcessor may not accept 'image_token' parameter")
    print("   - Some newer LLaVA features might not be available")
    print("   - The code has been updated to handle these limitations")
    
    print("\n📋 Next Steps:")
    print("1. If versions are compatible: python fix_tokenizer.py")
    print("2. Then: python test_model_loading.py")
    print("3. Finally: python run.py args/aokvqa.yaml")
    
    print("\n💡 If you continue having issues:")
    print("   - Try restarting the Colab runtime (Runtime → Restart runtime)")
    print("   - Clear browser cache and cookies")
    print("   - Use a fresh Colab notebook")

if __name__ == "__main__":
    main() 