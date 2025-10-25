"""
Example script showing how to use torch tao quantization with forge.

This script demonstrates three approaches:
1. Standalone quantization utility
2. Quantization for saving/deployment
3. Automatic quantization in forge pipeline
"""

import asyncio
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_standalone_quantization():
    """
    Example 1: Use the standalone quantization utility to quantize a model.
    
    This is useful for one-time quantization of a model before deployment.
    """
    from quantize import QuantizationConfig, quantize_model
    
    logger.info("=" * 80)
    logger.info("Example 1: Standalone Model Quantization")
    logger.info("=" * 80)
    
    # Load a model (using a small one for testing)
    model_name = "gpt2"  # Small model for testing
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create quantization config for int8
    config = QuantizationConfig(method="int8")
    logger.info(f"Quantization config: method={config.method}")
    
    # Quantize the model
    quantized_model = quantize_model(model, config, device="cpu")
    
    logger.info(f"Model quantized successfully!")
    logger.info(f"Original model size: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Quantized model size: {quantized_model.get_memory_footprint() / 1e6:.2f} MB")
    
    return quantized_model


def example_quantize_and_save():
    """
    Example 2: Quantize a model and save it for later use.
    
    Useful for preprocessing models before deployment to save time during initialization.
    """
    from quantize import quantize_and_save, QuantizationConfig
    
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Quantize and Save Model")
    logger.info("=" * 80)
    
    model_name = "gpt2"
    output_path = "quantized_model.pt"
    
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure quantization with int4
    config = QuantizationConfig(
        method="int4",
        group_size=128,
    )
    logger.info(f"Quantization config: method={config.method}, group_size={config.group_size}")
    
    # Quantize and save
    quantize_and_save(model, output_path, config=config, device="cpu")
    
    file_size = Path(output_path).stat().st_size / 1e6
    logger.info(f"Saved quantized model to {output_path} ({file_size:.2f} MB)")


async def example_forge_pipeline_quantization():
    """
    Example 3: Use quantization in forge pipeline with Generator.
    
    This shows how to integrate quantization into the forge Actor workflow.
    Quantization happens automatically after model loading.
    """
    from forge.actors import Generator, ForgeQuantizationConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Quantization in Forge Pipeline")
    logger.info("=" * 80)
    
    # Create quantization configuration (disabled by default for safety)
    quant_config = {
        "enabled": True,  # Set to True to enable quantization
        "method": "int8",
        "group_size": None,
    }
    logger.info(f"Quantization config: {quant_config}")
    
    # Create engine configuration
    engine_args = EngineArgs(
        model="gpt2",  # Small model for testing
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_num_seqs=1,
        max_seq_len_to_capture=4096,
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )
    
    logger.info("Spawning Generator with quantization...")
    
    try:
        # Spawn generator with quantization config
        generator = await Generator.options(
            procs=1,
            num_replicas=1,
            with_gpus=False,  # Set to True if GPU available
        ).as_service(
            engine_args=engine_args,
            sampling_params=sampling_params,
            quantization_config=quant_config,
        )
        
        # Generate text
        logger.info("Generating text...")
        prompt = "The future of AI is"
        results = await generator.generate(prompt)
        
        logger.info(f"Prompt: {prompt}")
        for result in results:
            logger.info(f"Generated: {result.text}")
        
        # Cleanup
        await generator.shutdown()
        
    except Exception as e:
        logger.error(f"Error in forge pipeline: {e}")
        raise


def main():
    """Run all examples."""
    logger.info("Torch TAO Quantization Examples")
    logger.info("=" * 80)
    
    # Example 1: Standalone quantization
    try:
        quantized_model = example_standalone_quantization()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
    
    # Example 2: Quantize and save
    try:
        example_quantize_and_save()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
    
    # Example 3: Forge pipeline (async)
    # Note: This requires a working forge setup with GPU support
    # Uncomment to test with actual forge deployment
    # try:
    #     asyncio.run(example_forge_pipeline_quantization())
    # except Exception as e:
    #     logger.error(f"Example 3 failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
