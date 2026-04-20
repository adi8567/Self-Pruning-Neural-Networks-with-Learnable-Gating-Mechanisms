import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.network import PrunableNet
from utils.loss import sparsity_loss
from utils.metrics import calculate_sparsity
import config

def train(lambda_val):
    device = config.DEVICE

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.CIFAR10(
    root='./data',   # path where you placed folder
    train=True,
    download=False,  # ❗ important
    transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    model = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()

        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Add sparsity loss
            s_loss = sparsity_loss(model)
            total = loss + lambda_val * s_loss

            total.backward()
            optimizer.step()

            total_loss += total.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    sparsity = calculate_sparsity(model)
    print(f"Sparsity: {sparsity:.2f}%")

    return model