import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function

# 1. Define the Deep Neural Network
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        # Input is 28x28 = 784
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10) # Output has 10 classes (digits 0-9)

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# 2. Set up device, data, and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.0001

# 3. Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. Initialize model, loss, and optimizer
model = DeepNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Train the model with Profiler
print("\nStarting training...")
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/mnist_dnn'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Step the profiler
            prof.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Finished training.")

# 6. Print profiler results to the console
print("\n" + "="*80)
print("PROFILER RESULTS")
print("="*80)

# 6.1. Sort by CPU time
print("\n--- Top 10 Operations by CPU Time ---")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 6.2. Sort by CUDA time (if using GPU)
if torch.cuda.is_available():
    print("\n--- Top 10 Operations by CUDA Time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 6.3. Sort by memory usage
print("\n--- Top 10 Operations by Memory Usage ---")
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# 6.4. Group by input shapes
print("\n--- Operations Grouped by Input Shape (Top 10 by CPU Time) ---")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

# 6.5. Trace export
# NOTE: on_trace_ready already writes Chrome traces to ./log/mnist_dnn.
# Calling export_chrome_trace here would raise "Trace is already saved."
print("\n--- Chrome traces saved to: ./log/mnist_dnn ---")

# 6.6. Export stacks for flame graph (optional)
# Uncomment if you want to generate flame graphs
# prof.export_stacks("profiler_stacks.txt", "self_cpu_time_total")

# 7. Test the model
print("\nStarting evaluation...")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

print("\n" + "="*80)
print("PROFILING COMPLETE")
print("="*80)

"""
ALTERNATIVE: Using cudaProfilerStart/Stop with Nsight Systems
--------------------------------------------------------------
If you want to use NVIDIA Nsight Systems for profiling, replace the
torch.profiler.profile() context manager with:

import torch

# Only profile specific sections of code
torch.cuda.cudart().cudaProfilerStart()

# ... your training code here ...

torch.cuda.cudart().cudaProfilerStop()

Then run your script with:
    nsys profile -o output_profile python pytorch_profiler.py

View the results:
    nsys stats output_profile.nsys-rep
    # Or open output_profile.nsys-rep in Nsight Systems GUI

This approach is useful for:
- Detailed GPU kernel analysis
- Multi-GPU profiling
- System-wide performance analysis
- Integration with other NVIDIA tools
"""
