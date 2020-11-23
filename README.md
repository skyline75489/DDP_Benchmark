DDP Benchmark
=============

The script is based on the one in PyTorch official repo: https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/ddp/benchmark.py

The procedure is to take a pretrained model, do 15 iterations (5 for warmup + 10 for benchmark) of fine-tuning with random data. The loss will be recorded by tensorboard.

### Requirements:

* tensorboard
* tensorboardx
* [transformers (for BERT)](https://github.com/huggingface/transformers)

*Note*: You need to modify the script manually for it to be able to run all the NNs. I know. It's not good.

To run the benchmark:

```plaintext
python benchmark.py --world-size 2 --master-addr localhost --master-port 9000 --distributed-backend gloo --rank 0
```

To view the loss statistics:

```plaintext
tensorboard --logdir=runs
```