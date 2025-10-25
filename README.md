# TorchAO Quantization for vLLM

Team Name: Segfault

This project extends vLLM with a weight transfer scheme that supports TorchAO quantization.

Recently, vLLM merged a [PR that allows for weight quantization with TorchAO](https://github.com/vllm-project/vllm/pull/23014). This PR requires a full snapshot of the weight tensor to exist on the inference side. If you're using TorchForge for RDMA weight streaming, then such a snapshot never exists.

Our first implementation performs a gather-all across the shards to create a snapshot that is then quantized. This is memory-intensive and wasteful as all GPUs / nodes are idle during accumulation and quantization. Additionally, a large memory snapshot is created and sent at once which can be latency insensitive.

We provide an additional implementation that attempts to avoid the gather-all. Trainers running on TorchTitan quantize and prepack model weights on-GPU, shard them by tensor-parallel rank, aggregate to compute global quantization metrics, and stream the result shards directly to vLLM inference workers over TorchForge RDMA or TCP.

## Description

In a weight transfer between GPU nodes of vLLM node and RL trainer node, it needs an all-to-all comm cause sharding layouts are different. If vLLM is quantised (int8) but trainer is not (fp16), getting precise quants on vLLM needs computing global stats (mean, var) -- global over shards.

Normally, if the Trainer is TP, we send $1$ message to concat all Trainer tensors, then send $1$ message to send the Full tensor (over RMDA) to vLLM (and vLLM needs the full tensor for resharding). Even in a B200, data movement is the biggest bottleneck, and each message send is high data traffic between devices (GPUs to $1$ GPU, then $1$ GPU to other GPUs).

To reduce waste, we want to use fast CPU RAM as a double-ended buffer, so all trainer workers flush their tensor shards to RAM in parallel. The trainers are immediately freed to work on the next iteration in async, and the vLLM workers read from RAM. For quantization, we cache local statistics (mean, var) per shard on the trainer side, then the vLLMs aggregate statistics (i.e. mean of means) over shards.

## Discord Usernames

* Neel Somani (@neelsomani)
* Arihan Varanasi (@ar1v)
* Markus Zhang (@photonmz)
* Aryan Bansal (@imaginedragons0519)
