# TorchAO Quantization for vLLM

Team Name: Segfault

This project extends vLLM with a weight transfer scheme that supports TorchAO quantization.

Recently, vLLM merged a [PR that allows for weight quantization with TorchAO](https://github.com/vllm-project/vllm/pull/23014). This PR requires a full snapshot of the weight tensor to exist on the inference side. If you're using TorchForge for RDMA weight streaming, then such a snapshot never exists.

Our first implementation performs a gather-all across the shards to create a snapshot that is then quantized. This is memory-intensive and wasteful as all GPUs / nodes are idle during accumulation and quantization. Additionally, a large memory snapshot is created and sent at once which can be latency insensitive.

We provide an additional implementation that attempts to avoid the gather-all. Trainers running on TorchTitan quantize and prepack model weights on-GPU, shard them by tensor-parallel rank, aggregate to compute global quantization metrics, and stream the result shards directly to vLLM inference workers over TorchForge RDMA or TCP.

## Discord Usernames

* Neel Somani (@neelsomani)
* Arihan Varanasi (@ar1v)
* Markus Zhang (@photonmz)