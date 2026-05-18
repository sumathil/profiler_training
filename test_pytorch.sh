#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
source /home/vader/Desktop/Agentic_AI_finance/.venv_gpu/bin/activate
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

python examples/pytorch/mnist_cnn.py --epochs 1 --batch-size 128 --max-steps 200
./scripts/profile_nsys_mnist.sh nsys_mnist 1 128 200 ./data
python examples/pytorch/cifar10_resnet_amp.py --epochs 1 --batch-size 256 --max-steps 200 --channels-last --use-profiler --profiler-chrome-trace

python examples/pytorch/cifar10_resnet_amp.py --epochs 1 --batch-size 256 --max-steps 200 --channels-last
./scripts/profile_nsys_cifar10_resnet.sh nsys_cifar10_resnet 1 256 200 ./data


python examples/pytorch/mnist_cnn.py --epochs 1 --batch-size 128 --max-steps 200 --use-profiler
python examples/pytorch/cifar10_resnet_amp.py --epochs 1 --batch-size 256 --max-steps 200 --channels-last --use-profiler

python examples/pytorch/mnist_cnn.py --epochs 1 --batch-size 128 --max-steps 200 --use-profiler --profiler-chrome-trace
python examples/pytorch/cifar10_resnet_amp.py --epochs 1 --batch-size 256 --max-steps 200 --channels-last --use-profiler --profiler-chrome-trace
