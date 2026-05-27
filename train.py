import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from eval.metrics import evaluate_clean
from models.simple_cnn import SimpleCNN


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch} "
                f"[{batch_idx * len(images)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-path", type=str, default="simple_cnn_mnist.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        clean_acc = evaluate_clean(model, test_loader, device)
        print(f"Clean Acc: {clean_acc * 100:.2f}%")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
