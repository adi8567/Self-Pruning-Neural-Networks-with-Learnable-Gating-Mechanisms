import torch
import torchvision
import torchvision.transforms as transforms
from models.network import PrunableNet
from utils.metrics import calculate_sparsity
import config

def evaluate(model):
    device = config.DEVICE

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")