import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import sys

# Check if ROCm/HIP is available
if not torch.cuda.is_available():
    print("Error: CUDA/ROCm not available. Please check your PyTorch installation.")
    print("For ROCm: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
    sys.exit(1)

# Print device information
device = torch.device('cuda')
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Device properties: {torch.cuda.get_device_properties(0)}")

# Note: ROCm profiling is done externally via rocprof
# No need to import profiling libraries in the code

# Simple neural network for MNIST
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def load_data(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Acc: {100. * correct / total:.2f}%')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


def profile_model(model, device, sample_input, num_iterations=100):
    """Profile model inference"""
    model.eval()
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    torch.cuda.synchronize()

    # Measure time
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    with torch.no_grad():
        for i in range(num_iterations):
            start_events[i].record()
            _ = model(sample_input)
            end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)

    print(f"\nInference profiling ({num_iterations} iterations):")
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Throughput: {1000.0 / avg_time:.1f} samples/sec (batch_size=1)")


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    print("\n=== MNIST Training on ROCm ===\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data(batch_size)

    # Create model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    print("\n=== Starting Training ===\n")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, device, train_loader,
                                           criterion, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 60)

    # Profile inference
    print("\n=== Profiling Inference ===")
    sample_input = torch.randn(1, 1, 28, 28)
    profile_model(model, device, sample_input)

    # Save model
    model_path = 'mnist_rocm_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    print("\n=== Training Complete ===")
    print("\nTo profile this script with rocprof:")
    print("  rocprof --stats --hip-trace python mnist_rocm.py")
    print("\nTo generate timeline trace:")
    print("  rocprof --sys-trace --hip-trace python mnist_rocm.py")
    print("  # View results.json in chrome://tracing")


if __name__ == '__main__':
    main()
