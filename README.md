
# modelling-unigram-priv

[![CircleCI](https://circleci.com/gh/tpimentelms/modelling-unigram-priv.svg?style=svg&circle-token=dd03e792e49ec51ee6d7cedb1f01e2271ca9739b)](https://circleci.com/gh/tpimentelms/modelling-unigram-priv)

Modelling the unigram distribution code


## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch:
```bash
$ conda install pytorch torchvision cpuonly -c pytorch
$ python -m spacy download xx_ent_wiki_sm
```

## Running the code

The code can be run using the command `make`. The language to use is defined in the `Makefile`.

`get_wiki`: Download and preprocess wikipedia data.

`train`: Train the model.

`eval`: Train using both types and tokens and evaluate the models on the test set. Writes results to the results folder.

## Contributing

Please run `pylint src/ --rcfile .pylintrc` in the root folder to run the linter.