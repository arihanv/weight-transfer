import torch
import forge
import vllm

print(f'PyTorch: {torch.__version__}')
print(f'TorchForge: {forge.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')