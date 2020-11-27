DDP Benchmark
=============

The script is based on the one in PyTorch official repo: https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/ddp/benchmark.py

The procedure is to take a pretrained model, do 15 iterations (5 for warmup + 10 for benchmark) of fine-tuning with random data. The loss will be recorded by tensorboard.

### Requirements:

* PyTorch & torchvision
* tensorboard
* tensorboardx
* [transformers (for BERT)](https://github.com/huggingface/transformers)

### Installation:

```plaintext
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tensorboard tensorboardx transformers
```


To run the benchmark:

```plaintext
python imagenet_main.py smaller --arch resnet50 --world-size 1 --multiprocessing-distributed --dist-backend gloo --dist-url file:///D:\pg --rank 0 --batch-size 128
```

To view the loss statistics:

```plaintext
tensorboard --logdir=runs
```
