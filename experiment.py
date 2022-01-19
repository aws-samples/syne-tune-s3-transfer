import argparse
import json
import tempfile
import time
from pathlib import Path
import os

import boto3
from boto3.s3.transfer import TransferConfig
from syne_tune.report import Reporter

s3 = boto3.resource("s3")


# HyperBand needs checkpointing functions
# those are not used for the bayes_fifo search

def load_checkpoint(checkpoint_path: Path) -> dict:
    state = {'num': 0, 'cumsum': 0.0}
    try:
        if checkpoint_path is not None:
            with open(checkpoint_path, "r") as f:
                state = json.load(f)
    except Exception:
        pass
    return state


def save_checkpoint(checkpoint_path: Path, state: dict):
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path.parent, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(state, f)
            
            
def run_one_download(bucket, key, file_name, file_path, config):
    """generic S3 download function to invoke in experiments"""
    
    # write with tempfile to avoid overwrites
    with tempfile.TemporaryDirectory(dir=file_path) as local_path:
        t1 = time.time()
        s3.Object(bucket_name=bucket, key=key).download_file(
            Filename=str(Path(local_path) / file_name), Config=config
        )
        duration = time.time() - t1

    return duration



parser = argparse.ArgumentParser()


if __name__ == "__main__":
    
    # non-searchable config
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--n_downloads", type=int, required=True)
    parser.add_argument("--search", type=str, required=True)
    
    # convention - Syne Tune serialize/deserialize path (for Hyperband)
    parser.add_argument('--st_checkpoint_dir', type=str)

    # search space
    parser.add_argument("--max_concurrency", type=int, required=True)
    parser.add_argument("--multipart_chunksize", type=int, required=True)
    parser.add_argument("--max_io_queue", type=int, required=True)
    parser.add_argument("--io_chunksize", type=int, required=True)
    
    args, _ = parser.parse_known_args()
    
    
    # call the Reporter here: creation timestamp documents worker start
    report = Reporter()
    
    config = TransferConfig(
        max_concurrency=args.max_concurrency,
        multipart_chunksize=args.multipart_chunksize,
        max_io_queue=args.max_io_queue,
        io_chunksize=args.io_chunksize,
    )
    
    # experiment (averaged over N fresh downloads to reduce jitter)
    
    if args.search == "bayes_fifo":
    
        durations = []
        
        for _ in range(args.n_downloads):
            
            duration = run_one_download(
                bucket=args.bucket,
                key=args.key,
                file_name=args.file_name,
                file_path=args.file_path,
                config=config)
                        
            durations.append(duration)
        
        avg_download_time = sum(durations)/len(durations)
        report(avg_download_time=avg_download_time)
        

    elif args.search == "bayes_hyperband":
        
        # Hyperband runs experiment by chunks, to evaluate which one to continue
        if args.st_checkpoint_dir is not None:
            checkpoint_path = Path(args.st_checkpoint_dir) / "checkpoint.json"
        else:
            checkpoint_path = None
        state = load_checkpoint(checkpoint_path)
    
        for _ in range(state['num'], args.n_downloads):
            
            duration = run_one_download(
                bucket=args.bucket,
                key=args.key,
                file_name=args.file_name,
                file_path=args.file_path,
                config=config)
    
            state['num'] += 1
            state['cumsum'] += duration
            download_attempts = state['num']
            avg_download_time = state['cumsum'] / download_attempts
            
            # unlike FIFO where we report final experiment metrics (avg duration)
            # in Hyperband we report itermediary steps too, for further optimization
            report(download_attempts=download_attempts, avg_download_time=avg_download_time)
            save_checkpoint(checkpoint_path, state)

        
        