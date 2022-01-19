import argparse
import logging

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.search_space import lograndint, randint
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

parser = argparse.ArgumentParser()

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    

    # non-searchable config
    parser.add_argument("--bucket", type=str, required=True,
                        help="S3 bucket where the benchmarked file is located")
    
    parser.add_argument("--key", type=str, required=True,
                        help="S3 key of the benchmarked file")
    
    parser.add_argument("--file_name", type=str, required=True,
                        help="Name to give to the benchmarked file when locally downloaded")
    
    parser.add_argument("--file_path", type=str, required=True,
                        help="Local path to write the benchmarked file to")
    
    parser.add_argument("--init", type=str, default="random",
                        choices=["random", "boto3_defaults"],
                        help="start searching from a random config or from boto3_defaults")
    
    parser.add_argument("--n_downloads", type=int, default=3,
                        help="Number of downloads per experiment to reduce outliers impact")
    
    parser.add_argument("--search", type=str, default="bayes_hyperband",
                        choices=["bayes_hyperband", "bayes_fifo"],
                        help="The search backend. bayes_hyperband supports early stopping")    
    
    parser.add_argument("--max_tuning_time", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    
    
    args, _ = parser.parse_known_args()
    
    
    config_space = {
        "bucket": args.bucket,
        "key": args.key,
        "file_name": args.file_name,
        "file_path": args.file_path,
        "n_downloads": args.n_downloads,
        "search": args.search,
        "max_concurrency": randint(10, 1000),
        "max_io_queue": lograndint(1e2, 1e5),
        "io_chunksize": lograndint(1e3, 1e9),
        "multipart_chunksize": lograndint(1e6, 1e10),
    }
    
    # Syne Tune lets us specify a search starting point. 
    if args.init == "boto3_defaults":  # 01/22 Boto3 default
        baseline = [{
            "max_concurrency": 10,
            "max_io_queue": 100,
            "io_chunksize": 262144,
            "multipart_chunksize": 8388608,
        }]
        
    elif args.init == "random":
        baseline = None
    
    
    # Syne Tune scheduler config shared across both schedulers
    searcher="bayesopt"
    mode='min'
    metric='avg_download_time'

    if args.search == "bayes_fifo":
    
        scheduler = FIFOScheduler(
            config_space,
            random_seed=args.seed,
            points_to_evaluate=baseline,            
            searcher=searcher,  
            mode=mode,
            metric=metric,
        )

    if args.search == "bayes_hyperband":
        
        # 'HyperbandScheduler' runs asynchronous successive halving, or Hyperband.
        # It starts a trial whenever a worker is free.
        # The 'promotion' variant pauses each trial at certain resource levels
        # (called rungs). Trials which outperform others at the same rung, are
        # promoted later on, to run to the next higher rung.
        # We configure this scheduler with Bayesian optimization: configurations
        # for new trials are selected by optimizing an acquisition function based
        # on a Gaussian process surrogate model. The latter models learning curves
        # f(x, r), x the configuration, r the number of epochs done, not just final
        # values f(x).
        
        config_space.update({
            'max_resource_attr': args.n_downloads,
        })
        
        search_options = {
        'num_init_random': 3,
        'gp_resource_kernel': 'matern52'
        }
        
        scheduler = HyperbandScheduler(
            config_space,
            random_seed=args.seed,
            points_to_evaluate=baseline,            
            searcher=searcher,  
            mode=mode,
            metric=metric,            
            type='promotion',
            search_options=search_options,
            rung_levels=list(range(1, args.n_downloads + 1)),
            resource_attr='download_attempts',  # identifies experiment iterations
            max_resource_attr='n_downloads',
        )
        
    
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_tuning_time)
    
    # We will run the various experiments on the local machine (same as the tuner)
    backend = LocalBackend(entry_point="experiment.py")
    
    # Launch the tuner
    # In this S3 local download tuning we use only 1 worker to 
    # avoid inter-experiment contention.    
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
    )
    
    print(f"results saved at {tuner.tuner_path}")
    print(f"S3 download time measured by averaging {args.n_downloads} downloads")
    
    tuner.run()