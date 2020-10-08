import click
import math
import torch
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader
from data import DatasetFromFolder
from models import SRCNN
import torch.nn as nn


@click.group()
def cli():
    pass

@cli.command()
@click.option('--zoom', type=int, required=True)
@click.option('--epochs', type=int, default=100)
def train(zoom, epochs):
    #specifying hardware accleration
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    #seed to produce reproducible results
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)

    #defining parameters
    BATCH_SIZE = 4
    NUM_WORKERS = 0

    trainset = DatasetFromFolder("data/train", zoom_factor=zoom)
    testset = DatasetFromFolder("data/test", zoom_factor=zoom)
    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = SRCNN().to(device)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params":model.conv1.parameters(), "lr": 0.0001},
            {"params":model.conv2.parameters(), "lr": 0.0001},
            {"params":model.conv3.parameters(), "lr": 0.00001},
        ], lr=0.00001,
    )

    for epoch in range(epochs):
        epoch_loss= 0
        for i, batch in enumerate(trainloader):
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(input)
            loss = loss_func(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")

        avg_psnr = 0
        with torch.no_grad():
            for batch in testloader:
                input, target = batch[0].to(device), batch[1].to(device)

                out = model(input)
                loss = loss_func(out, target)
                psnr = 10 * log10(1 / loss.item())
                avg_psnr += psnr
        print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

    # Save model
        torch.save(model, f"model_{epoch}.pth")

if __name__ == '__main__':
    cli()
