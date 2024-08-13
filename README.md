# MyNanoGPT

MyNanoGPT is a lightweight implementation of the GPT2 architecture that I built to learn how it works. This is heavily inspired by Andrej Karpathy's nanogpt.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Generating Text](#generating-text)
- [Contributing](#contributing)
- [License](#license)

## Features

- Lightweight and easy to understand implementation of GPT-2.
- Supports training on custom datasets.
- Text generation capabilities with adjustable parameters.
- Built with PyTorch for flexibility and performance.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mynanogpt.git
   cd mynanogpt
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv myenv
   source myenv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use MyNanoGPT, you can either train the model on your dataset or use a pre-trained model.

### Training

To train the model, run the following command:
```
python train_gpt2.py
```
Make sure to configure the training parameters in the `config.py` file according to your needs.
