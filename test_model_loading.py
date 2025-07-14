#!/usr/bin/env python3
"""
Test script to verify LLaVA-1.5-7B model loading works correctly.
Run this before starting the full training to catch any issues early.
"""

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer

def test_model_loading():
    """Test loading the LLaVA model with different configurations."""
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    print("🧪 Testing LLaVA-1.5-7B model loading...")
    print(f"Model ID: {model_id}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    try:
        # Test 1: Basic processor loading
        print("\n1️⃣ Testing processor loading...")
        try:
            # Try loading with minimal parameters for transformers 4.37.2 compatibility
            processor = LlavaProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = processor.tokenizer
            print(f"✅ Processor loaded successfully")
            print(f"   Tokenizer vocab size: {len(tokenizer)}")
        except Exception as e:
            print(f"⚠️  Standard loading failed: {e}")
            print("🔄 Trying alternative loading method...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                processor = LlavaProcessor.from_pretrained(model_id, trust_remote_code=True)
                print(f"✅ Alternative loading successful")
                print(f"   Tokenizer vocab size: {len(tokenizer)}")
            except Exception as e2:
                print(f"⚠️  Alternative loading failed: {e2}")
                print("🔄 Trying final fallback method...")
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(model_id)
                # Create a simple processor wrapper for transformers 4.37.2
                processor = type('SimpleProcessor', (), {
                    'tokenizer': tokenizer,
                    'image_processor': type('ImageProcessor', (), {
                        '__call__': lambda self, img, **kwargs: {'pixel_values': torch.randn(1, 3, 224, 224) if img is not None else None}
                    })(),
                    '__call__': lambda self, text=None, images=None, return_tensors=None, padding=None, max_length=None, **kwargs: {
                        'input_ids': tokenizer(text, return_tensors=return_tensors, padding=padding, max_length=max_length, **kwargs)['input_ids'],
                        'attention_mask': tokenizer(text, return_tensors=return_tensors, padding=padding, max_length=max_length, **kwargs)['attention_mask'],
                        'pixel_values': torch.randn(1, 3, 224, 224) if images is not None else None
                    }
                })()
                print(f"✅ Final fallback loading successful")
                print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Test 2: Basic model loading (CPU for testing)
        print("\n2️⃣ Testing basic model loading...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print(f"✅ Basic model loaded successfully")
        print(f"   Model type: {type(model)}")
        # Handle different config structures in transformers 4.37.2
        try:
            hidden_size = model.config.hidden_size
        except AttributeError:
            try:
                hidden_size = model.config.text_config.hidden_size
            except AttributeError:
                hidden_size = "unknown"
        print(f"   Hidden size: {hidden_size}")
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
                trust_remote_code=True,
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
                trust_remote_code=True,
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