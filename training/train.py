import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn_model import DeepfakeCNN
from training.dataset import get_dataloaders

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders()

    model = DeepfakeCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):

        model.train()
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed")

    torch.save(model.state_dict(), "saved_models/best_model.pth")


if __name__ == "__main__":
    train()