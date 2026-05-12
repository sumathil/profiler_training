import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from contextlib import nullcontext

# Try to import nvtx (pynvtx library with color support)
try:
    import nvtx
    NVTX_AVAILABLE = True
    
    class nvtx_range:
        def __init__(self, msg, color="blue"):
            self.msg = msg
            self.color = color
            
        def __enter__(self):
            nvtx.push_range(self.msg, color=self.color)
            return self
            
        def __exit__(self, *args):
            nvtx.pop_range()
            
except ImportError:
    print("Warning: nvtx not available. Install with: pip install nvtx")
    NVTX_AVAILABLE = False
    nvtx_range = lambda msg, color=None: nullcontext()


# Define color scheme
class Colors:
    DATA = "green"
    FORWARD = "blue"
    BACKWARD = "red"
    OPTIMIZER = "yellow"
    LOSS = "orange"
    EVAL = "purple"
    EPOCH = "cyan"


# 1. Define the Deep Neural Network
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


# 2. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 3
batch_size = 100
learning_rate = 0.0001

# 3. Load MNIST Dataset
with nvtx_range("Data Loading", color=Colors.DATA):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, 
                                               transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, 
                                              transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. Initialize model
with nvtx_range("Model Initialization", color="white"):
    model = DeepNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Train the model
print("\nStarting training...")
if torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()

with nvtx_range("Training Loop", color="white"):
    for epoch in range(num_epochs):
        with nvtx_range(f"Epoch {epoch+1}/{num_epochs}", color=Colors.EPOCH):
            for i, (images, labels) in enumerate(train_loader):
                with nvtx_range(f"Iteration {i+1}", color="white"):
                    # Data transfer
                    with nvtx_range("Data Transfer (H2D)", color=Colors.DATA):
                        images = images.to(device)
                        labels = labels.to(device)
                    
                    # Forward pass
                    with nvtx_range("Forward Pass", color=Colors.FORWARD):
                        outputs = model(images)
                    
                    # Loss computation
                    with nvtx_range("Loss Computation", color=Colors.LOSS):
                        loss = criterion(outputs, labels)
                    
                    # Backward pass
                    with nvtx_range("Backward Pass", color=Colors.BACKWARD):
                        optimizer.zero_grad()
                        loss.backward()
                    
                    # Optimizer step
                    with nvtx_range("Optimizer Step", color=Colors.OPTIMIZER):
                        optimizer.step()
                
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")

if torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStop()

print("Finished training.")

# 6. Test the model
print("\nStarting evaluation...")

with nvtx_range("Evaluation", color=Colors.EVAL):
    with torch.no_grad():
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            with nvtx_range(f"Eval Batch {batch_idx+1}", color=Colors.EVAL):
                with nvtx_range("Data Transfer (H2D)", color=Colors.DATA):
                    images = images.to(device)
                    labels = labels.to(device)
                
                with nvtx_range("Forward Pass", color=Colors.FORWARD):
                    outputs = model(images)
                
                with nvtx_range("Prediction", color=Colors.EVAL):
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

print("\n" + "=" * 80)
print("PROFILING COMPLETE")
print("=" * 80)