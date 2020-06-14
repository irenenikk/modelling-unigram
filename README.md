
# Modelling the Unigram distribution

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

In order to change the hyperparameters of the two-stage model, define `ALPHA` and `BETA`.

You can see the individual steps of the experiments in the `Makefile`, but `all` will download the data (if it has not been downloaded already) and train the type and token models (if they have not been trained), retrain a two-stage model with the given hyperparameters, evaluate all models, and calculate type entropies as well as expected code lengths.

The results are stored under their own folder for each language in `results` by default.

You can also run the components of the experiments individually:

`get_wiki`: Download and preprocess wikipedia data.

`train_generator`: Train the generator LSTM language model.

`train_two_stage`: Train the two stage model consisting of a generator and an adaptor. Requires `train_generator` to be finished. The two-stage model is initialised both with a generator trained on tokens and one trained on types.

`eval_generator`: Evaluate the different initialisations of the generator language model. This will evaluate both the type and token models, as well as the retrained generators.

`eval_two_stage`: Evaluate the different initialisations of the two-stage model.

`calculate_suprisal`: Calculate the surprisal for individual types in the test set.

`calculate_average_sentence_length`: Calculate the average sentence lenghts under different models.

`tune_hyperparams`: Tune the hyperparameters for a given language. You can specify the amount of parameters tested with `TUNING_ITERATIONS` (the default is 10).

In evaluation the models are evaluated both using types and tokens.

## Reading the results

// TODO

## Contributing

Please run `pylint src/ --rcfile .pylintrc` in the root folder to run the linter.
