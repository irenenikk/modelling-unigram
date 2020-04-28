
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

In order to run the full experiments pipeline for a given language, run

```
make all LANGUAGE=<wikipedia language code> MAX_TRAIN_TOKENS=<the amount of tokens to use in training>
```

You can also run the components of the experiments individually:

`get_wiki`: Download and preprocess wikipedia data.

`train_generator`: Train the generator LSTM language model.

`train_two_stage`: Train the two stage model consisting of a generator and an adaptor. Requires `train_generator` to be finished. The two-stage model is initialised both with a generator trained on tokens and one trained on types.

`eval_generator`: Evaluate the different initialisations of the generator language model.

`eval_two_stage`: Evaluate the different initialisations of the two-stage model.

In evaluation the models are evaluated both using types and tokens.

## Contributing

Please run `pylint src/ --rcfile .pylintrc` in the root folder to run the linter.