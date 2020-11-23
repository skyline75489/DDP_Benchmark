#!/usr/bin/env python3
#
# Measure distributed training iteration time.
#
# This program performs a sweep over a) a number of model architectures, and
# b) an increasing number of processes. This produces a 1-GPU baseline,
# an 8-GPU baseline (if applicable), as well as measurements for however
# many processes can participate in training.
#

import argparse
import io
import itertools
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification, BertTokenizer

if not torch._six.PY3:
    raise RuntimeError("DDP benchmark requires Python 3")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def allgather_object(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    input_tensor = torch.ByteTensor(list(buffer.getvalue()))
    input_length = torch.IntTensor([input_tensor.size(0)])
    dist.all_reduce(input_length, op=dist.ReduceOp.MAX)
    input_tensor.resize_(input_length[0])
    output_tensors = [
        torch.empty(input_tensor.size(), dtype=torch.uint8)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(output_tensors, input_tensor)
    output = []
    for tensor in output_tensors:
        buffer = io.BytesIO(np.asarray(tensor).tobytes())
        output.append(torch.load(buffer))
    return output


def allgather_run(cmd):
    proc = subprocess.run(shlex.split(cmd), capture_output=True)
    assert(proc.returncode == 0)
    return allgather_object(proc.stdout.decode("utf-8"))



def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def benchmark_process_group(pg, benchmark, use_ddp_for_single_rank=True):
    torch.manual_seed(pg.rank())
    torch.cuda.manual_seed(pg.rank())

    model = benchmark.create_model()
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        0.001,
        momentum=0.9,
        weight_decay=1e-4)
    if use_ddp_for_single_rank or pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size)

    measurements = []
    warmup_iterations = 5
    measured_iterations = 10

    enable_tensorboard = pg.rank() == 0
    if enable_tensorboard:
        writer = SummaryWriter(comment='_' + benchmark.model + '_' + benchmark.prefix)
    iter = 0

    for (inputs, target) in (data * (warmup_iterations + measured_iterations)):
        iter = iter + 1
        start = time.time()
        if isinstance(benchmark, TorchvisionBenchmark):
            output = model(*inputs)
            loss = criterion(output, target)
        if isinstance(benchmark, TransformersBenchmark):
            encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            outputs = model(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'), labels=target)
            loss = outputs.loss

        if enable_tensorboard:
            writer.add_scalar("loss/iter", loss, iter)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        measurements.append(time.time() - start)

    if enable_tensorboard:
        writer.flush()
        writer.close()
    # Throw away measurements for warmup iterations
    return measurements[warmup_iterations:]


def run_benchmark(benchmark, ranks, opts):
    group = dist.new_group(ranks=ranks, backend=benchmark.distributed_backend)
    measurements = []
    if dist.get_rank() in set(ranks):
        if not opts:
            opts = dict()
        measurements = benchmark_process_group(group, benchmark, **opts)
    dist.destroy_process_group(group)
    dist.barrier()

    # Aggregate measurements for better estimation of percentiles
    return list(itertools.chain(*allgather_object(measurements)))


def sweep(benchmark):
    # Synthesize the set of benchmarks to run.
    # This list contain tuples for ("string prefix", [rank...]).
    benchmarks = []

    def append_benchmark(prefix, ranks, opts=None):
        prefix = "%4d GPUs -- %s" % (len(ranks), prefix)
        benchmarks.append((prefix, ranks, opts))

    def local_print(msg):
        if dist.get_rank() == 0:
            print(msg, end='', flush=True)  # noqa: E999

    def print_header():
        local_print("\n")
        local_print("%22s" % "")
        for p in [50, 75, 90, 95]:
            local_print("%14s%10s" % ("sec/iter", "ex/sec"))
        local_print("\n")

    def print_measurements(prefix, nelem, measurements):
        measurements = sorted(measurements)
        local_print("%8s:" % prefix)
        for p in [50, 75, 90, 95]:
            v = np.percentile(measurements, p)
            local_print("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))
        local_print("\n")

    # Every process runs once by themselves to warm up (CUDA init, etc).
    append_benchmark("  warmup", [dist.get_rank()], {"use_ddp_for_single_rank": False})

    # Single machine baselines
    append_benchmark("  no ddp", range(1), {"use_ddp_for_single_rank": False})
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    #append_benchmark("   1M/4G", range(4))

    # Multi-machine benchmarks
    #for i in range(1, (dist.get_world_size() // 8) + 1):
    #    append_benchmark("   %dM/8G" % i, range(i * 8))

    # Run benchmarks in order of increasing number of GPUs
    print_header()
    results = []
    for prefix, ranks, opts in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # Turn range into materialized list.
        ranks = list(ranks)
        benchmark.prefix = prefix
        measurements = run_benchmark(benchmark, ranks, opts)
        if "warmup" not in prefix:
            print_measurements(prefix, benchmark.batch_size, measurements)
            results.append({"ranks": ranks, "measurements": measurements})

    return results


class Benchmark(object):
    def __init__(self, device, distributed_backend, bucket_size):
        self.device = device
        self.batch_size = 32
        self.distributed_backend = distributed_backend
        self.bucket_size = bucket_size

    def __str__(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def generate_inputs(self):
        raise NotImplementedError

    def generate_target(self):
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    def __init__(self, device, distributed_backend, bucket_size, model):
        super(TorchvisionBenchmark, self).__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model

    def __str__(self):
        return "{} with batch size {}".format(self.model, self.batch_size)

    def create_model(self):
        return torchvision.models.__dict__[self.model]().to(self.device)

    def generate_inputs(self):
        return [torch.rand([self.batch_size, 3, 224, 224], device=self.device)]

    def generate_target(self):
        return torch.tensor([1] * self.batch_size, dtype=torch.long, device=self.device)


class TransformersBenchmark(Benchmark):
    def __init__(self, device, distributed_backend, bucket_size, model):
        super(TransformersBenchmark, self).__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model

    def __str__(self):
        return "{} with batch size {}".format(self.model, self.batch_size)

    def create_model(self):
        model = BertForSequenceClassification.from_pretrained(self.model, return_dict=True)
        model.train()
        return model.to('cuda')

    def generate_inputs(self):
        return [
        "President-elect Joe Biden has expedited the selection of his Cabinet and is planning to make the first of several key announcements next week, an official said, as part of a concerted effort to show that he is moving forward despite President Trump’s increasingly brazen attempts to sabotage the election.",
"Today, Biden said he has settled on his choice for Treasury secretary, but officials said he’s also reached a decision – or on the cusp of doing so – on other critical Cabinet posts, a few of which are expected to be announced before Thanksgiving.",
"Monday and Tuesday are being eyed as tentative days for the first introductions of members of Biden’s Cabinet, an official said, with others coming later.",
"Lael Brainard, a member of the Board of Governors of the Federal Reserve, is seen as the top contender to lead the Treasury Department.",
"If selected, she would become the first woman Treasury secretary, ", "a move that would help Biden to start to deliver on his pledge to name a diverse Cabinet.",
"But three officials close to the Biden transition declined to say whether Brainard was the final choice,", "saying it is a closely-held decision that Biden would likely reveal right after Thanksgiving.",
"But Biden could announce his choice for Secretary of State as soon as next week, officials said, along with another Cabinet post.",
"While Biden is well-known for his deliberate and often slow decision-making, particularly on personnel matters, the timeline of some Cabinet decisions is being accelerated because of a desire to move quickly to form a new government in the wake of Trump’s intransigence about the election.",
"Biden had talked with his advisers about taking a far slower approach", "including waiting for the outcome of the Georgia Senate run-offs that will determine control of the Senate, but Trump’s actions have motivated Biden to move without delay.",
"There is a desire to convey that we are governing, operating and up and running,” an official close to the transition said, explaining the urgency facing Biden’s team in the wake of Trump’s antics.",
"It’s been only a week since Ron Klain was named White House chief of staff,", "but that decision jumpstarted movement inside the Biden team. And Jeff Zients, a co-chair of the transition, has been working for months on a variety of options for Biden to make about top personnel decisions",
"When Donna Garrett was growing up in the greater Los Angeles area, it didn't seem strange that her mom worked as an airline pilot. Because her dad had the same profession, captaining an airplane seemed like a normal thing to do.",
"It was the boring job that my parents did when they went to work,", "Donna, now 26, laughs.",
"In fact, mom Suzy Garrett was blazing a trail across the skies as one of the first female pilots for regional US carrier SkyWest.",
"As Donna got older", "she began to take notice. Inspired by her parents' passion and the freedom they enjoyed to explore the world", "she decided to pursue her own career in flying.",
"Fast forward to September 2019 and Donna was operating an airplane alongside Suzy as SkyWest's first mother-daughter pilot team.",
"That flight took place over a year ago, but in recent weeks","Donna and Suzy's story has unexpectedly gone viral, with photos of the duo, smiling proudly in the cockpit, spreading across social media.",
"We knew it was really special,", "says Suzy, who was celebrating 30 years at SkyWest when she paired up with her daughter."
"She remembers ", "everybody else's reaction",  "as being one of the most heart-warming parts of the experience.",
"I was really surprised -- as surprised as now it's gone viral -- but even that day, I haven't had my picture taken that much since my wedding! Passengers taking pictures with us, rampers, flight attendants...That just helped make the day even more special, the support was really wonderful.",
"The pair had hoped to repeat the experience in 2020, but their plans were halted by the Covid pandemic.", "Right now, Suzy is in Los Angeles and Donna is in Chicago and, like many families, they've not been able to spend much time together this year."
]
    def generate_target(self):
        return torch.tensor([1] * self.batch_size, dtype=torch.long, device=self.device)

def main():
    parser = argparse.ArgumentParser(description='PyTorch distributed benchmark suite')
    parser.add_argument("--rank", type=int, default=os.environ.get("RANK", 0))
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--master-addr", type=str, required=False)
    parser.add_argument("--master-port", type=str, required=False)
    parser.add_argument("--model", type=str)
    parser.add_argument("--json", type=str, metavar="PATH", help="Write file with benchmark results")
    args = parser.parse_args()

    #num_gpus_per_node = torch.cuda.device_count()
    #assert num_gpus_per_node == 8, "Expected 8 GPUs per machine"

    # The global process group used only for communicating benchmark
    # metadata, like measurements. Not for benchmarking itself.
    if sys.platform == 'win32':
        init_method="file:///d:/Test/pg.txt"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=args.rank,
            world_size=args.world_size
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
            rank=args.rank,
            world_size=args.world_size,
        )

    #output = allgather_run("nvidia-smi topo -m")
    #if not allequal(output):
    #    print('Output of "nvidia-smi topo -m" differs between machines')
    #    sys.exit(1)

    if args.rank == 0:
        print("-----------------------------------")
        print("PyTorch distributed benchmark suite")
        print("-----------------------------------")
        print("")
        print("* PyTorch version: {}".format(torch.__version__))
        print("* CUDA version: {}".format(torch.version.cuda))
        print("* Distributed backend: {}".format(args.distributed_backend))
        print("* Maximum bucket size: {}MB".format(args.bucket_size))
        print("")
        print("--- nvidia-smi topo -m ---")
        print("")
        #print(output[0])
        print("--------------------------")
        print("")

    torch.cuda.set_device(dist.get_rank() % 8)
    device = torch.device('cuda:%d' % (dist.get_rank() % 8))

    benchmarks = []
    if args.model:
        benchmarks.append(
            TorchvisionBenchmark(
                device=device,
                distributed_backend=args.distributed_backend,
                bucket_size=args.bucket_size,
                model=args.model))
    else:
        for model in ["bert-base-uncased"]:#, "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            benchmarks.append(
                TransformersBenchmark(
                    device=device,
                    distributed_backend=args.distributed_backend,
                    bucket_size=args.bucket_size,
                    model=model))

    benchmark_results = []
    for benchmark in benchmarks:
        if args.rank == 0:
            print("\nBenchmark: {}".format(str(benchmark)))
        result = sweep(benchmark)
        benchmark_results.append({
            "model": benchmark.model,
            "batch_size": benchmark.batch_size,
            "result": result,
        })

    # Write file with benchmark results if applicable
    if args.rank == 0 and args.json:
        report = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "distributed_backend": args.distributed_backend,
            "bucket_size": args.bucket_size,
            "benchmark_results": benchmark_results,
        }
        with open(args.json, 'w') as f:
            json.dump(report, f)


if __name__ == '__main__':
    main()
