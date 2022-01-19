# S3 Download Tuner

## What is this? 
In this example, we use the open-source parameter tuning library Syne Tune to learn an optimal configuration of `boto3`'s `Download_file` method, for a given file and a given client EC2 instance type. Using this code, we were able to reduce the download time of various files (5GiB, 10GiB) up to ~40% vs boto3 default transfer configuration. Consider this repo as a one-off creative demo for Syne Tune. 


## What is Syne Tune?
Syne Tune is a distributed parameter search library open-sourced by AWS AI researchers. It integrates several state-of-the art optimization concepts:
 * Can handle local workers or remote, asynchronous workers to run many experiments asynchronously in parallel
 * Comes with latest optimization innovations such as MoBster (https://arxiv.org/pdf/2003.10865.pdf), population-based search and HyperBand (https://arxiv.org/abs/1603.06560)
 * Supports advanced features such as multi-objective search and constraints.
 
While it has its roots in machine learning and machine learning task optimization, Syne Tune is use-case agnostic and can be used for generic complex system tuning, beyond machine learning: it explores and finds arbitrary, user-defined configuration that optimizes a user-defined metric. 
 
Read more here:
 * https://aws.amazon.com/blogs/machine-learning/run-distributed-hyperparameter-and-neural-architecture-tuning-jobs-with-syne-tune/
 * https://github.com/awslabs/syne-tune
 

## How to run this demo?

To get started, open the Jupyter notebook `Demo.ipynb` and follow the instructions

At any time after the tuner is launched, you can open and run the notebook `Evaluation.ipynb` to check the status of the parameter search.

To make the demo portable, we wrap the Syne Tune tuning code in a Python script that can be invoked the following way:

```
    python launcher.py \
        --bucket $bucket \
        --key $key \
        --file_path $file_path \
        --file_name $file_name \
        --init random \
        --max_tuning_time 10800
```

Our script takes the following arguments:

* `bucket`: S3 bucket storing the file whose download speed you want to optimize
* `key`: S3 key of the file whose download speed you want to optimize
* `file_path`: local path to download the file to
* `file_name`: local name to download the file to
* `init`: how to start the Syne Tune search. Valid values are `random` (try a random configuration first) and `boto3_defaults` (start from S3 boto3 Transfer configuration defaults). Defaults to `random`
* `n_downloads`: How many times to test a single download configuration. S3 download duration can have variations (for example caused by network conditions). To reduce the impact of outliers, we optimize the average of N downloads. Defaults to 3. 
* `max_tuning_time`: maximum search duration in seconds. Defaults to `3600`.
* `search`: the search logic, powered by Syne Tune schedulers and searchers. Valid values are `bayes_fifo` and `bayes_hyperband`. Defaults to `bayes_hyperband`. `bayes_fifo` runs full sets of N downloads per configuration, and uses a bayesian optimization search to find the next configuration to try. `bayes_hyperband` runs downloads one by one across many configuration, and over-invests on configuration that behave well ; it can early-stop experimenting on bad configurations. Hence `bayes_hyperband` is expected to perform generally faster than `bayes_fifo`. We leave the code for `bayes_fifo` in the demo, because it is simpler to port and understand, and because Hyperband is relevant only for experiments emitting a timeseries of metrics (like our N donwload times). Note than Syne Tune offers much more possibilities in terms of search, and we implemented only two of them in this demo. Syne Tune documentation contains more information about the schedulers https://github.com/awslabs/syne-tune/blob/main/docs/schedulers.md
* `seed`: random seed for Syne Tune. Defaults to `42`


## Areas of improvement
Here are ideas to search faster - we may add those improvements in the future

1. Parallelism: run the experiments on remote transient workers (such as SageMaker jobs), so that several configurations can be tested in parallel

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

