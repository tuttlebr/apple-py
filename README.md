# Accelerated PyTorch Training on Mac

In collaboration with the Metal engineering team at Apple, we are excited to announce support for GPU-accelerated PyTorch training on Mac. Until now, PyTorch training on Mac only leveraged the CPU, but with the upcoming PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training. This unlocks the ability to perform machine learning workflows like prototyping and fine-tuning locally, right on Mac.

## METAL ACCELERATION

Accelerated GPU training is enabled using Apple’s Metal Performance Shaders (MPS) as a backend for PyTorch. The MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. MPS optimizes compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family. The new device maps machine learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.

## TRAINING BENEFITS ON APPLE SILICON

Every Apple silicon Mac has a unified memory architecture, providing the GPU with direct access to the full memory store. This makes Mac a great platform for machine learning, enabling users to train larger networks or batch sizes locally. This reduces costs associated with cloud-based development or the need for additional local GPUs. The Unified Memory architecture also reduces data retrieval latency, improving end-to-end performance.

In the graphs below, you can see the performance speedup from accelerated GPU training and evaluation compared to the CPU baseline:

![perf data](https://pytorch.org/assets/images/METAPT-002-BarGraph-02.gif)

- Testing conducted by Apple in April 2022 using production Mac Studio systems with Apple M1 Ultra, 20-core CPU, 64-core GPU 128GB of RAM, and 2TB SSD. Tested with macOS Monterey 12.3, prerelease PyTorch 1.12, ResNet50 (batch size=128), HuggingFace BERT (batch size=64), and VGG16 (batch size=64). Performance tests are conducted using specific computer systems and reflect the approximate performance of Mac Studio.

## GET STARTED LOCALLY

1. Build local python env.

   `python3 -m venv apple-py`

2. Activate local python env.

   `source apple-py/bin/activate`

3. Install requirements.

   `pip3 install -r requirements.txt`

4. Follow the example training and inference [readme](workspace/super_resolution/README.md).
