# BP-GPT
Code of the paper BP-GPT: Auditory Neural Decoding Using fMRI-prompted LLM.

## Training

### Downloading

1. Download the [semantic decoding dataset](https://github.com/HuthLab/semantic-decoding).

2. Download the [BERT](https://huggingface.co/google-bert/bert-base-uncased) and [GPT-2](https://huggingface.co/openai-community/gpt2).


### Text2text

1. Modify the config in configs/config.py, train the text2text model by setting stage=1.

### Brain2text

1. Train the brain2text model by setting stage=2.


## Acknowledgements

Some parts of the code of this project are adapted from [semantic-decoding](https://github.com/HuthLab/semantic-decoding) and [CLIP-prefix-caption](https://github.com/rmokady/CLIP_prefix_caption). Special thanks to their great works.