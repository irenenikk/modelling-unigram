
# Modelling the Unigram distribution

[![CircleCI](https://circleci.com/gh/tpimentelms/modelling-unigram-priv.svg?style=svg&circle-token=dd03e792e49ec51ee6d7cedb1f01e2271ca9739b)](https://circleci.com/gh/tpimentelms/modelling-unigram-priv)


Code for modelling the unigram distribution using a Pitman--Yor process and a character-level LSTM.

## Dependencies

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch:
```bash
$ conda install pytorch torchvision cpuonly -c pytorch
$ python -m spacy download xx_ent_wiki_sm
```

The code includes a copy of [this Wikipedia tokenizer](https://github.com/tpimentelms/wiki-tokenizer) with the consent of the creator.

## Running the code

In order to run the full experiments pipeline for a given language, run

```
make all LANGUAGE=<wikipedia language code> MAX_TRAIN_TOKENS=<the amount of tokens to use in training>
```

In order to change the hyperparameters of the two-stage model, define `ALPHA` and `BETA`.

You can see the individual steps of the experiments in the `Makefile`, but `all` will download the data (if it has not been downloaded already) and train the type and token models (if they have not been trained), retrain a two-stage model with the given hyperparameters, evaluate all models, and calculate type entropies as well as expected code lengths.

The results are stored under their own folder for each language in `results` by default.
Two-stage models are distinguished between each other with the following naming convention: `two_stage_init_<generator init dataset>_<value of hyperparameter a>_<value of hyperparameter b>_<max number of training tokens>` for the two-stage model, `tokens_<max number of training tokens>` for the LSTM trained on token data and `types_<max number of training tokens>` for the LSTM trained on type data.

You can also run the components of the experiments individually:

`get_wiki`: Download and preprocess wikipedia data.

`train_generator`: Train the generator LSTM language model.

`train_two_stage`: Train the two stage model consisting of a generator and an adaptor. Requires `train_generator` to be finished. The two-stage model is initialised both with a generator trained on tokens and one trained on types.

`eval_generator`: Evaluate the different initialisations of the generator language model. This will evaluate both the type and token models, as well as the retrained generators. These results are stored into `lstm_results`.

`eval_two_stage`: Evaluate the different initialisations of the two-stage model.

`calculate_suprisal`: Calculate the surprisal for individual types in the test set. These are stored into a file named `entropy_freq.csv`.

`calculate_average_sentence_length`: Calculate the average sentence lenghts under different models. Results are saved into `average_sentence_lengths`.

`tune_hyperparams`: Tune the hyperparameters for a given language. You can specify the amount of parameters tested with `TUNING_ITERATIONS` (the default is 10). The results are stored in a CSV file called `hyperparam_tuning_results`.

In evaluation the models are evaluated both using types and tokens.
## Contributing

Please run `pylint src/ --rcfile .pylintrc` in the root folder to run the linter.
