FROM python

WORKDIR /workspace

COPY worker.py /workspace/
COPY alexnet_cifar10.py /workspace/
COPY file /workspace/file

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python3", "worker.py"]