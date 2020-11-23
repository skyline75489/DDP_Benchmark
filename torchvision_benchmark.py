import os
import inspect
import argparse
import time
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def get_data_loaders(train_batch_size, val_batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return trainloader


def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def train(pg, model, device, model_name, backend):
    # params you need to specify:
    epochs = 5
    train_loader = get_data_loaders(256, 256)
    loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    optimizer = optim.Adadelta(model.parameters())

    start_ts = time.time()

    batches = len(train_loader)

    enable_tensorboard = pg.rank() == 0
    if enable_tensorboard:
        writer = SummaryWriter(comment='_' + model_name + '_' + backend)

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # ----------------- TRAINING  --------------------
        # set model to training
        model.train()

        measurements = []
        for i, data in enumerate(train_loader):
            start = time.time()
            X, y = data[0].to(device), data[1].to(device)

            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            torch.cuda.synchronize()
            duration = time.time() - start
            print(str(duration) + ' ' + str(total_loss))
            measurements.append(duration)

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        writer.add_scalar("total loss", total_loss, epoch)

    print(f"Training time: {time.time()-start_ts}s")


def main():
    parser = argparse.ArgumentParser(description='PyTorch distributed benchmark suite')
    parser.add_argument("--rank", type=int, default=os.environ.get("RANK", 0))
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--model", type=str)
    parser.add_argument("--json", type=str, metavar="PATH", help="Write file with benchmark results")
    args = parser.parse_args()

    init_method="file:///d:/Test/pg"

    # initialize the global process group
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=args.rank,
        world_size=args.world_size
    )

    MODEL_NAME = args.model
    BUCKET_SIZE = 25

    device = torch.device('cuda:%d' % (dist.get_rank() % 8))
    model = torchvision.models.__dict__[MODEL_NAME](False).to(device)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)

    pg = dist.new_group(ranks=range(args.world_size), backend=args.distributed_backend)

    if pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size)

    train(pg, model, device, MODEL_NAME, args.distributed_backend)

if __name__ == '__main__':
    main()
