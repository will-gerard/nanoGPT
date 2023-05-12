
# Optimized Transfer Learning on GPT for Automatic Short Answer Grading

This repository is based on the original NanoGPT repository: https://github.com/karpathy/nanoGPT, from which it is forked.

![nanoGPT](assets/nanogpt.jpg)

## Project Description
In this work we extend the open source NanoGPT repository to perform transfer learning on a pretrained GPT model to complete the Automatic Short Answer Grading task. By reducing the pretrained model to a simpler architecture with fewer parameters, optimizing the data loading portion of the training loop, timing and profiling the code, and switching to more powerful hardware, we are able to reduce runtime on the ASAG task without harming signifanctly harming performance. Our final results fall short of state of the art, however our framework can now be used to easily experiment with new architectures and settings to try to improve ASAG performance.

## Code Outline

## Sample Commands
To perform the transfer learning related operations, please ensure you are in an environment with everything in the Requirements.txt file installed. Also note that in order to actually perform one of these transfer learning operations, an appropriate checkpoint file from a pretraining job must exist in the pretrained/ directory. If you would like to recreate experiments detailed below and in our paper, please reach out to one of the authors of this repo, who can provide you with a checkpoint, or explain how to create a new one.

With everything in place, different functionalities can be exercised as explained below:

Preparing OpenWebText dataset 

```
python data/openwebtext/prepare.py
```

Pretraining:

```
python train.py config/train_12_768_1024.py
```

Transfer Learning Training:

```
python train.py config/transfer_train.py
```

Evaluation:

```
python eval_transfer.py
```

Hyperparameter Sweep:

```
python hyperparam_sweep.py
```

## Results



