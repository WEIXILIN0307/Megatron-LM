"""Exercise real finetune.get_batch and TP broadcast under torchrun."""
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import httpx
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoTokenizer

import finetune
from megatron.core import mpu
from modelopt.torch.speculative.plugins.hf_streaming_dataset import (
    EagleVllmStreamingConfig,
    MegatronEagleVllmStreamingDataset,
)


@record
def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=20))
    mpu.initialize_model_parallel(tensor_model_parallel_size=2)
    finetune.get_args = lambda: SimpleNamespace(
        eagle_streaming_server_urls=os.environ['PROBE_SERVER'],
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    finetune.get_run_config = lambda: SimpleNamespace(
        model=SimpleNamespace(
            transformer=SimpleNamespace(max_seq_length_per_sample=64),
            transformations=SimpleNamespace(original_num_layers=28),
        )
    )
    original = finetune.tensor_parallel.broadcast_data

    def broadcast(*args, **kwargs):
        print(f'rank={rank} entering real TP broadcast at {time.monotonic():.3f}', flush=True)
        return original(*args, **kwargs)

    finetune.tensor_parallel.broadcast_data = broadcast
    iterator = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(os.environ['PROBE_TOKENIZER'], local_files_only=True)
        ds = MegatronEagleVllmStreamingDataset(
            [{'conversation_id': 'timeout', 'messages': [
                {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}
            ]}],
            tokenizer,
            EagleVllmStreamingConfig(
                server_urls=os.environ['PROBE_SERVER'], model='test', max_seq_len=64,
                request_timeout=1, fail_after_consecutive_skips=1,
            ),
        )

        def prepare_transport():
            # Descriptor/RDMA transfer is never reached: the HTTP request times out.
            ds._http_rdma = httpx.Client(timeout=1)
            ds._rr = 0

        ds._rdma = prepare_transport
        iterator = iter(torch.utils.data.DataLoader(ds, batch_size=1))
    torch.distributed.barrier(device_ids=[rank])
    print(f'rank={rank} starting batch at {time.monotonic():.3f}', flush=True)
    finetune.get_batch(iterator)
    raise AssertionError('failed sample unexpectedly produced a batch')


if __name__ == '__main__':
    main()
