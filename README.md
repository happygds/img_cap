# Self-critical Sequence Training for Image Captioning

We download the dataset from the official webset and utilize the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention.git) repository to obtain image features. 
Then we modified the [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch.git) repository to train the caption model.

## Requirements
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)
PyTorch 0.2 (along with torchvision)

## Extracing bottom-up features
Firstly, we download pretrained model for extaction by following the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention.git). 
The official code hasn't been modified since we only utilize it for extraction.
Thus we run the `bottom-up-attention/generate_bufeats_pi.py` to extract features, where we obtain 64 features per image(fixed).
Here, to uitilize the gpu resources of cloud and reduce extraction time, we equally divide the dataset and submit multiple extraction tasks, which is indicated by the `splits` parameter.

## Pretraining glove vectors based on the dataset captions
We find that using glove vectors pretrained from the dataset captions would be a little beneficial.
Run the `self-critical.pytorch/data/prepro_glove.py` to convert the captions of dataset to 'text8' file.
Then use the official [GloVe](https://github.com/stanfordnlp/GloVe.git) repository to train glove vectors.

## Train the captioning network
The prepocessing of dataset could follow the original instructions.
We made some modifications to the [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch.git) repository, as follows:
- The preprocessing code is modified, mainly including the `scripts/*.py` and `data/*.py` files.
- The `opts.py/train.py` is modified for command-line manner, for example:
```
python opts.py --learning_rate 4e-4 --start_from 'save/c2f_glove' --temperature 1. --seed 1024 --checkpoint_path 'save/c2f_glove' --seq_length 25 --batch_size 16 --id 17
```
- We implement the [stack-captioning](https://arxiv.org/abs/1709.03376) model and add the corresponding `class C2FTopDownCore()` method in `models/AttModel.py`.
- We train the model with different seeds to obtain mutiple models for ensemble. 
Thus we add the ensemble code for evaluation in `models/AttModel.py`, including the 'sample_beam_emsemble() / beam_search_emsemble(0) / get_logprobs_state_emsemble()' methods.
