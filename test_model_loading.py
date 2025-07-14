#!/usr/bin/env python3
"""
Test script to verify LLaVA-1.5-7B model loading works correctly.
Run this before starting the full training to catch any issues early.
"""

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig

def test_model_loading():
    """Test loading the LLaVA model with different configurations."""
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    print("🧪 Testing LLaVA-1.5-7B model loading...")
    print(f"Model ID: {model_id}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    try:
        # Test 1: Basic processor loading
        print("\n1️⃣ Testing processor loading...")
        processor = LlavaProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
        print(f"✅ Processor loaded successfully")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Test 2: Basic model loading (CPU for testing)
        print("\n2️⃣ Testing basic model loading...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print(f"✅ Basic model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test 3: Add special tokens
        print("\n3️⃣ Testing special token addition...")
        special_tokens_dict = {
            "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Special tokens added successfully")
        print(f"   New vocab size: {len(tokenizer)}")
        print(f"   Latent token ID: {tokenizer.convert_tokens_to_ids('<|latent|>')}")
        
        # Test 4: Test with GPU if available
        if torch.cuda.is_available():
            print("\n4️⃣ Testing GPU loading...")
            device = torch.device("cuda")
            model = model.to(device)
            print(f"✅ Model moved to GPU successfully")
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Test 5: Test basic forward pass
        print("\n5️⃣ Testing basic forward pass...")
        test_text = "Hello, how are you?"
        inputs = processor(text=test_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {outputs.logits.shape}")
        
        print("\n🎉 All tests passed! Model loading is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_options():
    """Test loading with optimization options."""
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    print("\n🔧 Testing optimization options...")
    
    try:
        # Test 4-bit quantization (if bitsandbytes is available)
        try:
            from transformers import BitsAndBytesConfig
            print("\n   Testing 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print("   ✅ 4-bit quantization works")
            
        except ImportError:
            print("   ⚠️  bitsandbytes not available, skipping 4-bit test")
        
        # Test Flash Attention 2 (if available)
        try:
            print("\n   Testing Flash Attention 2...")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                use_flash_attention_2=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("   ✅ Flash Attention 2 works")
            
        except Exception as e:
            print(f"   ⚠️  Flash Attention 2 not available: {e}")
        
        print("\n✅ Optimization tests completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during optimization testing: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LLaVA-1.5-7B Model Loading Tests")
    print("=" * 50)
    
    # Run basic tests
    basic_success = test_model_loading()
    
    # Run optimization tests
    opt_success = test_optimization_options()
    
    print("\n" + "=" * 50)
    if basic_success:
        print("🎉 Basic model loading tests PASSED")
        print("✅ You can proceed with training!")
    else:
        print("❌ Basic model loading tests FAILED")
        print("⚠️  Please fix the issues before training")
    
    if opt_success:
        print("✅ Optimization tests completed")
    else:
        print("⚠️  Some optimization options may not be available")
    
    print("\n📝 Next steps:")
    print("1. If all tests passed, run: python run.py args/aokvqa.yaml")
    print("2. If tests failed, check your environment and dependencies")
    print("3. Make sure you have sufficient GPU memory (16GB+ recommended)") 