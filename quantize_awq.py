import os
import warnings
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Suppress non-critical kernel fallback warnings
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# Import vLLM official llm-compressor framework

try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
except ImportError:
    oneshot = None

def merge_and_quantize_llm_compressor(
    base_model_name: str = "Qwen/Qwen3.5-4B-Base",
    lora_model_name: str = "pngwn/system-one-qwen3.5-4b-scorer-v2b",
    merged_output_dir: str = "./system-one-merged",
    quantized_output_dir: str = "./system-one-4bit-awq",
    num_calibration_samples: int = 128,
    max_seq_length: int = 512
):
    print("=========================================================================")
    print(" General-Purpose System One W4A16 4-bit Quantization Workflow")
    print("=========================================================================")

    # Step 1: Merge LoRA Adapter into Base Model (or load pre-merged model locally)
    if os.path.exists(os.path.join(merged_output_dir, "config.json")):
        print(f"\n--- Loading pre-merged model from local directory '{merged_output_dir}' ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(merged_output_dir, local_files_only=True)
            merged_model = AutoModelForCausalLM.from_pretrained(
                merged_output_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        except Exception as e:
            print(f"Could not load local merged model directly ({e}). Attempting remote load...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            lora_model = PeftModel.from_pretrained(base_model, lora_model_name)
            merged_model = lora_model.merge_and_unload()
            os.makedirs(merged_output_dir, exist_ok=True)
            merged_model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
    else:
        print(f"\n--- Step 1: Merging Base Model '{base_model_name}' + LoRA '{lora_model_name}' ---")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        lora_model = PeftModel.from_pretrained(base_model, lora_model_name)
        merged_model = lora_model.merge_and_unload()

        os.makedirs(merged_output_dir, exist_ok=True)
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Merged model saved to '{merged_output_dir}'.")

    # Step 2: Quantize using llm-compressor with Hugging Face Calibration Dataset
    print(f"\n--- Step 2: Quantizing Merged Model with Hugging Face Dataset Calibration (W4A16) ---")
    if oneshot is None:
        print("\nERROR: 'llmcompressor' is not installed.")
        print("Please install llm-compressor via: pip install llm-compressor torchvision datasets")
        return

    # Standard Hugging Face calibration dataset registered in llmcompressor
    hf_dataset_name = "ultrachat-200k"
    print(f"Using registered calibration dataset '{hf_dataset_name}' ({num_calibration_samples} samples, max_len={max_seq_length})...")

    # Define GPTQ calibration-based quantization recipe
    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
        recipe = [
            GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"],
                dampening_frac=0.01
            )
        ]
        print("Using GPTQModifier with second-order error compensation and calibration data...")
    except ImportError:
        recipe = [
            QuantizationModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"]
            )
        ]
        print("Using standard QuantizationModifier with calibration dataset pass...")

    # Run oneshot quantization pass with Hugging Face calibration dataset
    oneshot(
        model=merged_model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=hf_dataset_name,
        splits="train_sft",
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        output_dir=quantized_output_dir
    )



    tokenizer.save_pretrained(quantized_output_dir)
    print(f"\n=========================================================================")
    print(f" SUCCESS! Calibrated W4A16 4-bit Model saved to '{quantized_output_dir}'.")
    print("=========================================================================")




if __name__ == "__main__":
    merge_and_quantize_llm_compressor()

