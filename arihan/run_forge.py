#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forge RL loop for Qwen with checkpointing + LATEST pointer + (optional) quantization.

What it does:
- Starts:
  * DatasetActor (GSM8K stream, CPU)
  * Policy service (GPU) for online generations
  * RLTrainer (GPU) for parameter updates
  * ReplayBuffer (CPU)
  * ComputeAdvantages (CPU)
  * ReferenceModel (GPU, frozen baseline)
  * RewardActor service (CPU) for MathReward/ThinkingReward

- Training loop:
  dataset → policy.generate → reward → advantages → buffer → trainer.step

- Every N steps:
  trainer.save_checkpoint() → publish LATEST pointer
  policy hot-reloads from LATEST (so serving uses newest weights)
  (optional) quantize checkpoint and publish separate LATEST for vLLM

To keep this script drop-in, the quantizer is optional. If your Forge build doesn’t
include quant actors, it will skip quantization gracefully.
"""

import asyncio
import os
import sys
from typing import List

# --- Forge imports (as in your snippet) ---
from forge.actors.generator import Generator as Policy
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reference_model import ReferenceModel
from forge.actors.trainer import RLTrainer
from apps.grpo.main import DatasetActor, RewardActor, ComputeAdvantages
from forge.data.rewards import MathReward, ThinkingReward

# Optional quantizers: guarded import; skip if unavailable
QuantizerAWQ = None
QuantizerGPTQ = None
try:
    # If your Forge build exposes quant actors under forge.actors.quant
    from forge.actors.quant import QuantizerAWQ as _QA, QuantizerGPTQ as _QG
    QuantizerAWQ = _QA
    QuantizerGPTQ = _QG
except Exception:
    pass

# -----------------------
# CONFIG (edit these)
# -----------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")  # or "Qwen/Qwen3-1.7B"
USE_BF16 = True

# Where to save raw checkpoints (HF layout). S3/GS/file paths all fine.
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "s3://forge-models/myrun")
LATEST_POINTER  = os.environ.get("LATEST_POINTER",  f"{CHECKPOINT_ROOT}/LATEST")

# Optional quantization & where to publish quantized model for vLLM
ENABLE_QUANT   = os.environ.get("ENABLE_QUANT", "false").lower() in ("1", "true", "yes")
QUANT_METHOD   = os.environ.get("QUANT_METHOD", "awq")  # "awq" | "gptq"
QUANT_ROOT     = os.environ.get("QUANT_ROOT",   "s3://forge-models/myrun-awq")
QUANT_LATEST   = os.environ.get("QUANT_LATEST", f"{QUANT_ROOT}/LATEST")

# Trainer loop knobs
TOTAL_OUTER_STEPS   = int(os.environ.get("TOTAL_OUTER_STEPS", "300"))  # how many rollout+train iters to run
CHECKPOINT_EVERY    = int(os.environ.get("CHECKPOINT_EVERY", "50"))
LOCAL_BATCH_SIZE    = int(os.environ.get("LOCAL_BATCH_SIZE", "4"))
SEQ_LEN             = int(os.environ.get("SEQ_LEN", "2048"))
LEARNING_RATE       = float(os.environ.get("LEARNING_RATE", "5e-6"))

# Policy decoding for rollouts
GROUP_SIZE      = int(os.environ.get("GROUP_SIZE", "1"))
MAX_GEN_TOKENS  = int(os.environ.get("MAX_GEN_TOKENS", "64"))
TEMP            = float(os.environ.get("TEMP", "0.8"))
TOP_P           = float(os.environ.get("TOP_P", "0.95"))

# Misc
PRINT_EVERY          = int(os.environ.get("PRINT_EVERY", "10"))
REPLAY_BATCH_SIZE    = int(os.environ.get("REPLAY_BATCH_SIZE", "4"))
REPLAY_MAX_POLICY_AGE= int(os.environ.get("REPLAY_MAX_POLICY_AGE", "2"))

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
async def save_and_publish_latest(trainer, pointer_uri: str) -> str:
    """Save a checkpoint and update the LATEST pointer. Returns the checkpoint dir URI."""
    ckpt_dir = await trainer.save_checkpoint()
    # Some Forge builds expose a publish_pointer helper on trainer; if not, emulate via your own writer.
    try:
        await trainer.publish_pointer(latest_uri=pointer_uri, target=ckpt_dir)
    except Exception:
        # Fallback: write a tiny JSON or text file via trainer helper if exposed, otherwise just print
        print(f"[warn] publish_pointer not available; newest ckpt: {ckpt_dir}", flush=True)
    return ckpt_dir

async def hot_reload_policy(policy, uri: str) -> None:
    """Ask the policy service to atomically swap weights from the given URI or pointer."""
    # Different Forge versions expose different names; try a few.
    for meth in ("update_weights", "reload", "load_weights"):
        fn = getattr(policy, meth, None)
        if fn is None:
            continue
        try:
            await fn(uri=uri)
            print(f"[policy] hot-reloaded weights from: {uri}", flush=True)
            return
        except TypeError:
            # Some APIs use `path` instead of `uri`
            try:
                await fn(path=uri)
                print(f"[policy] hot-reloaded weights from: {uri}", flush=True)
                return
            except Exception:
                pass
        except Exception as e:
            print(f"[policy] {meth} failed: {e}", flush=True)
    print("[policy] WARNING: no supported hot-reload method found; policy will keep old weights.", flush=True)

async def maybe_quantize_and_publish(ckpt_dir: str) -> str | None:
    """Optionally quantize the checkpoint and publish QUANT_LATEST. Returns the quantized dir or None."""
    if not ENABLE_QUANT:
        return None
    if QUANT_METHOD.lower() == "awq" and QuantizerAWQ:
        quant = await QuantizerAWQ.options(procs=1).as_actor(out_uri=QUANT_ROOT, awq_args={"max_seq_len": SEQ_LEN})
        out_dir = await quant.quantize(ckpt_dir)
        try:
            await quant.publish_pointer(QUANT_LATEST, out_dir)
        except Exception:
            print(f"[quant] AWQ done at {out_dir} (pointer publish skipped)", flush=True)
        print(f"[quant] AWQ quant published: {out_dir}", flush=True)
        return out_dir
    if QUANT_METHOD.lower() == "gptq" and QuantizerGPTQ:
        quant = await QuantizerGPTQ.options(procs=1).as_actor(out_uri=QUANT_ROOT, gptq_args={"bits": 4, "group_size": 128})
        out_dir = await quant.quantize(ckpt_dir)
        try:
            await quant.publish_pointer(QUANT_LATEST, out_dir)
        except Exception:
            print(f"[quant] GPTQ done at {out_dir} (pointer publish skipped)", flush=True)
        print(f"[quant] GPTQ quant published: {out_dir}", flush=True)
        return out_dir

    print("[quant] Skipping quantization (method unsupported or quant actor not available).", flush=True)
    return None

# ------------------------------------------------------
# Main program
# ------------------------------------------------------
async def main():
    print(f"[config] MODEL_ID={MODEL_ID}")
    print(f"[config] CHECKPOINT_ROOT={CHECKPOINT_ROOT}")
    if ENABLE_QUANT:
        print(f"[config] QUANT={QUANT_METHOD} → {QUANT_ROOT}")

    dtype = "bfloat16" if USE_BF16 else "float16"

    # Spin everything up in parallel
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        # Dataset actor (CPU)
        DatasetActor.options(procs=1).as_actor(
            path="openai/gsm8k",
            revision="main",
            data_split="train",
            streaming=True,
            model=MODEL_ID,
        ),
        # Policy service (GPU)
        Policy.options(procs=1, with_gpus=True, num_replicas=1).as_service(
            engine_config={
                "model": MODEL_ID,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,   # keep 1 unless you truly pipeline
                "enforce_eager": False,
            },
            sampling_config={
                "n": GROUP_SIZE,
                "max_tokens": MAX_GEN_TOKENS,
                "temperature": TEMP,
                "top_p": TOP_P,
            },
        ),
        # Trainer (GPU)
        RLTrainer.options(procs=1, with_gpus=True).as_actor(
            model={"name": "qwen", "flavor": "0.5B", "hf_assets_path": f"hf://{MODEL_ID}"},
            optimizer={"name": "AdamW", "lr": LEARNING_RATE},
            training={"local_batch_size": LOCAL_BATCH_SIZE, "seq_len": SEQ_LEN, "dtype": dtype},
            checkpoint={"folder": CHECKPOINT_ROOT, "interval": 0},  # we'll call save manually
        ),
        # Replay buffer (CPU)
        ReplayBuffer.options(procs=1).as_actor(
            batch_size=REPLAY_BATCH_SIZE,
            max_policy_age=REPLAY_MAX_POLICY_AGE,
            dp_size=1,
        ),
        # Advantage computation (CPU)
        ComputeAdvantages.options(procs=1).as_actor(),
        # Reference model (GPU) – frozen baseline
        ReferenceModel.options(procs=1, with_gpus=True).as_actor(
            model={"name": "qwen", "flavor": "0.5B", "hf_assets_path": f"hf://{MODEL_ID}"},
            training={"dtype": dtype},
        ),
        # Reward actor (CPU service)
        RewardActor.options(procs=1, num_replicas=1).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        ),
    )

    print("[startup] All actors/services are up.", flush=True)

    # --- Training Loop ---
    for step in range(1, TOTAL_OUTER_STEPS + 1):
        # 1) Fetch a small batch from dataset
        batch = await dataloader.next_batch()
        # Expect each item to have "question" – adjust if your dataset schema differs
        prompts: List[str] = [ex.get("question") or ex.get("prompt") or "" for ex in batch]

        # 2) Generate with the current policy (routed to least-loaded replica; microbatched)
        out = await policy.generate.route(prompts=prompts)
        # Normalize to list of strings
        if hasattr(out, "__iter__"):
            texts = []
            for x in out:
                # Completion objects often have .text; handle dicts too
                if hasattr(x, "text"):
                    texts.append(x.text)
                elif isinstance(x, dict) and "text" in x:
                    texts.append(x["text"])
                else:
                    texts.append(str(x))
        else:
            texts = [str(out)]

        # 3) Score with Reward service
        rewards = await reward_actor.score(texts=texts)

        # 4) Compute advantages (e.g., GRPO/PG style)
        advs = await compute_advantages.compute(rewards=rewards)

        # 5) Add to ReplayBuffer
        await replay_buffer.add(samples=batch, outputs=texts, advantages=advs)

        # 6) If buffer has enough, train one step
        if await replay_buffer.size() >= 1:
            train_batch = await replay_buffer.pop()
            train_stats = await trainer.step(train_batch)
        else:
            train_stats = {}

        if step % PRINT_EVERY == 0:
            print(f"[step {step}] rewards={rewards} train_stats={train_stats}", flush=True)

        # 7) Periodic checkpoint → publish LATEST → hot-reload policy → (optional) quantize
        if step % CHECKPOINT_EVERY == 0:
            ckpt_dir = await save_and_publish_latest(trainer, LATEST_POINTER)
            await hot_reload_policy(policy, LATEST_POINTER)
            q_dir = await maybe_quantize_and_publish(ckpt_dir)
            if q_dir:
                print(f"[vLLM] Quantized LATEST ready at: {QUANT_LATEST}", flush=True)
            else:
                print(f"[vLLM] Raw LATEST ready at: {LATEST_POINTER}", flush=True)

    print("[done] Training loop finished.", flush=True)
    print(f"[info] Latest raw pointer: {LATEST_POINTER}", flush=True)
    if ENABLE_QUANT:
        print(f"[info] Latest quant pointer: {QUANT_LATEST}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ctrl-c] Exiting.", flush=True)
        sys.exit(0)
