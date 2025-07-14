#!/usr/bin/env python3
"""
Fix script for LLaVA tokenizer issues.
This script clears cached tokenizer files and tests the fix.
"""

import os
import shutil
from pathlib import Path

def clear_tokenizer_cache():
    """Clear cached tokenizer files that might be corrupted."""
    
    print("🧹 Clearing tokenizer cache...")
    
    # Common cache locations
    cache_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/tmp/huggingface_cache",
        "/content/.cache/huggingface/hub"
    ]
    
    cleared = False
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            print(f"📁 Found cache at: {cache_path}")
            try:
                # Look for LLaVA-related files
                for root, dirs, files in os.walk(cache_path):
                    for file in files:
                        if "llava" in file.lower() and file.endswith(".json"):
                            file_path = os.path.join(root, file)
                            print(f"🗑️  Removing: {file_path}")
                            os.remove(file_path)
                            cleared = True
            except Exception as e:
                print(f"⚠️  Could not clear cache at {cache_path}: {e}")
    
    if cleared:
        print("✅ Tokenizer cache cleared")
    else:
        print("ℹ️  No LLaVA cache files found to clear")
    
    return cleared

def test_fixed_loading():
    """Test the fixed tokenizer loading."""
    
    print("\n🧪 Testing fixed tokenizer loading...")
    
    try:
        from transformers import LlavaProcessor, LlamaTokenizer
        
        model_id = "llava-hf/llava-1.5-7b-hf"
        
        # Try the fixed loading method for transformers 4.37.2
        print("📝 Testing with minimal parameters...")
        processor = LlavaProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        print(f"✅ SUCCESS! Tokenizer loaded with vocab size: {len(tokenizer)}")
        return True
        
    except Exception as e:
        print(f"❌ Still failing: {e}")
        return False

if __name__ == "__main__":
    print("🔧 LLaVA Tokenizer Fix Script")
    print("=" * 40)
    
    # Clear cache
    clear_tokenizer_cache()
    
    # Test loading
    success = test_fixed_loading()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Tokenizer fix successful!")
        print("✅ You can now run: python test_model_loading.py")
    else:
        print("❌ Tokenizer fix failed")
        print("⚠️  Try restarting the Colab runtime and running again") 