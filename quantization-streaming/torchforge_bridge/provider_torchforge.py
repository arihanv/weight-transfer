from __future__ import annotations
import threading
import queue
import asyncio
from typing import Iterator, Optional
import torch

from .provider_base import ShardMeta, StreamRecord, StreamingWeightProvider

try:
    import torchstore
except ImportError as e:
    raise RuntimeError("TorchStore is required for networking. Install torchstore or set PYTHONPATH") from e


class TorchForgeSender:
    """Sender class for sending tensor data to a TorchStore endpoint."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._store_name = f"torchforge_{endpoint}"
        # Don't initialize here - do it in send() method
    
    def send(self, data: dict):
        """Send tensor data to the TorchStore endpoint.
        
        Args:
            data: Dictionary containing 'fqn', 'meta', 'tensor', and 'version_id' keys
        """
        async def _send_async():
            # Initialize TorchStore if not already done
            try:
                await torchstore.initialize(store_name=self._store_name)
            except Exception:
                pass  # Already initialized
            
            # Use the fqn as the key and store the entire data dict
            key = data.get('fqn', 'unknown')
            await torchstore.put(key, data, store_name=self._store_name)
        
        try:
            # Run the async call synchronously
            asyncio.run(_send_async())
        except Exception as e:
            print(f"Failed to send {data.get('fqn', 'unknown')}: {e}")
            raise
    
    def close(self):
        """Close the client connection."""
        try:
            # Note: TorchStore doesn't have a direct close method for individual clients
            # The store persists until shutdown
            pass
        except Exception:
            pass

class TorchForgeProvider(StreamingWeightProvider):
    def __init__(self, endpoint: str, device: torch.device | None = None):
        self.endpoint = endpoint
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._store_name = f"torchforge_{endpoint}"
        self._q: "queue.Queue[StreamRecord]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        async def _async_run():
            # Initialize TorchStore if not already done
            try:
                await torchstore.initialize(store_name=self._store_name)
            except Exception:
                pass  # Already initialized
            
            # Poll for new keys periodically
            last_keys = set()
            while not self._stop.is_set():
                try:
                    # Get all keys from the store
                    current_keys = await torchstore.keys(store_name=self._store_name)
                    current_keys_set = set(current_keys)
                    
                    # Find new keys
                    new_keys = current_keys_set - last_keys
                    
                    if new_keys:
                        print(f"Found {len(new_keys)} new keys: {list(new_keys)}")
                    
                    for key in new_keys:
                        try:
                            # Get the data for this key
                            data = await torchstore.get(key, store_name=self._store_name)
                            
                            # Process the data into a StreamRecord
                            fqn = data.get("fqn", key)
                            m = data.get("meta", {})
                            meta = ShardMeta(
                                global_shape=tuple(m.get("global_shape", [])),
                                shard_axis=int(m.get("shard_axis", 0)),
                                start=int(m.get("start", 0)),
                                length=int(m.get("length", 0)),
                                layout_info=dict(m.get("layout_info", {})),
                            )
                            t: torch.Tensor = data.get("tensor", torch.empty(0)).to(self.device, non_blocking=True)
                            ver = int(data.get("version_id", 0))
                            rec = StreamRecord(fqn=fqn, meta=meta, payload=t, qmeta={}, version_id=ver)
                            self._q.put(rec)
                            print(f"Processed key {key} successfully")
                            
                        except Exception as e:
                            print(f"Error processing key {key}: {e}")
                            continue
                    
                    last_keys = current_keys_set
                    
                    # Sleep for a short time before polling again
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in TorchForgeProvider polling: {e}")
                    await asyncio.sleep(1.0)
        
        # Run the async function in the thread
        asyncio.run(_async_run())

    def iter_stream(self) -> Iterator[StreamRecord]:
        while not self._stop.is_set():
            try:
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                continue

    def manifest(self) -> Optional[dict]:
        return None

    def close(self) -> None:
        self._stop.set()
        # Note: TorchStore doesn't have a direct close method for individual clients
        # The store persists until shutdown
