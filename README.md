# LLaMA Model Fine-Tuning with Ternary Quantization

This project focuses on fine-tuning a pre-trained LLaMA model using ternary quantization techniques. The goal is to optimize the model's performance while reducing its memory footprint.
This is my 1.58 BitNet implementation based on this paper: https://arxiv.org/abs/2402.17764

Basically when you generate the model - the model is blank and you need to train it. This is where I'm having the biggest issues - I still cant seem to get the training to work properly. 
I need help with this implementation. I've take this as far as I can with my knowledge but for some reason I cant get the training to work properly. 

I've been testing different parameter models against the implementation and this is what I've observed:

| Parameter Size  | Model Size (MB/GB) |
| ------------- | ------------- |
| 350M  | 72 MB  |
| 750M  | 753 MB  |
| 1B  | 753 MB  |
| 3B  | 753 MB  |
| 14B  | 753 MB  |
| 34B  | 753 MB  |
| 70B  | 753 MB  |
| 100B  | 753 MB  |
| 120B  | 753 MB  |
| 300B  | 753 MB  |

I was able to create these size models on my 96GB M2 Max Macbook Pro.
Just an FYI these scripts are specifically created to work on MPS with a CPU fallback. I'm hoping I can get help to get it working through MLX once we've fixed the finetuning / training issues. 

I also think that this isnt completely optimized for memory management and theres probably opportunity for that. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Ternary Quantization](#ternary-quantization)
- [Custom Gradient Checkpointing](#custom-gradient-checkpointing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)

## Installation
Git clone the repo

## Usage
1. Run new-model-architecture-creation.py
2. You'll be prompted for how many parameters you want your model to be. The script will create the model and save it in the same repo as where the files are saved
3. Once the model is created - to fine-tune the LLaMA model, use the `trainingv2.py` script with the appropriate command-line arguments:
   
```
python trainingv2.py --dataset <dataset_path> --model_path <model_path> --batch_size <batch_size> --num_epochs <num_epochs> --learning_rate <learning_rate> --output_dir <output_directory> --iters <num_iterations> --max_length <max_sequence_length> --grad_accum_steps <gradient_accumulation_steps>
```
- `dataset_path`: Path to the dataset file.
- `model_path`: Path to the pre-trained LLaMA model.
- `batch_size`: Batch size for training.
- `num_epochs`: Number of training epochs.
- `learning_rate`: Learning rate for the optimizer.
- `output_directory`: Output directory to save the fine-tuned model.
- `num_iterations`: Number of training iterations.
- `max_sequence_length`: Maximum sequence length for input tokens.
- `gradient_accumulation_steps`: Number of steps for gradient accumulation.

## Dataset

The dataset should be in one of the following formats: txt, json, jsonl. The `preprocess_dataset` function in `trainingv2.py` handles the preprocessing of the dataset based on its format.
Here is the format that the jsonl file should be formatted in :

`{"text": "This is an example for the model."}`
For example: 
`{"text": "<s>[INST] Create an array of length 5 which contains all even numbers between 1 and 10. [/INST]arr = [2, 4, 6, 8, 10]</s>"}`

`{"text": "<s>[INST] Formulate an equation to calculate the height of a triangle given the angle, side lengths and opposite side length. [/INST]Height of triangle = opposite side length * sin (angle) / side length</s>"}`

`{"text": "<s>[INST] Write a replace method for a string class which replaces the given string with a given set of characters.string = \"Hello World!\"\nreplace_with = \"Greetings!\" [/INST]def replace(self, replace_with):\n    new_string = \"\"\n    for char in self:\n        if char == \" \":\n            new_string += replace_with\n        else:\n            new_string += char\n    return new_string</s>"}`

`{"text": "<s>[INST] Create an array of length 15 containing numbers divisible by 3 up to 45. [/INST]arr = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]</s>"}`

## Model Architecture

The LLaMA model architecture is defined in `llama_model.py`. It consists of an embedding layer, multiple decoder layers, and a language model head. The model uses RMSNorm for normalization and applies rotary position embeddings to the attention mechanism.

## Ternary Quantization

Ternary quantization is applied to the model's weights to reduce memory usage. The `QuantizedEmbedding` and `BitLinear` classes in `llama_model.py` handle the quantization of the embedding layer and linear layers, respectively. The `quantize_tensor` function in `quantization_utils.py` performs the actual quantization.

## Custom Gradient Checkpointing

To reduce memory consumption during training, custom gradient checkpointing is implemented in `custom_gradient_checkpointing.py`. The `custom_checkpoint` function is used to checkpoint the forward pass and compute gradients during the backward pass.

## Training

The `train` function in `trainingv2.py` handles the training process. It iterates over the dataset in batches, computes the loss, and performs gradient accumulation. The model's parameters are updated using an optimizer.

## Evaluation

The `evaluate` function in `trainingv2.py` evaluates the model on a validation set. It computes the average loss over the validation batches.

## Saving and Loading Models

The `save_pretrained` method in the `LlamaModel` class saves the fine-tuned model to the specified output directory. It quantizes the model's weights and saves them in the safetensors format. The `load_pretrained` method loads a pre-trained model from the specified model path.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

