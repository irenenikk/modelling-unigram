LANGUAGE := yo
ALPHA := 0.5
BETA := 1
MAX_TRAIN_TOKENS := 700000
DATA_DIR_BASE := ./data
# used in hyperparameter tuning
NO_BETAS = 10
NO_ALPHAS = 10
EPOCHS = 5
DATA_DIR_LANG := $(DATA_DIR_BASE)/$(LANGUAGE)
WIKI_DIR := $(DATA_DIR_LANG)/wiki
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR_BASE)/$(LANGUAGE)
RESULTS_DIR_BASE := ./results
RESULTS_DIR_LANG := $(RESULTS_DIR_BASE)/$(LANGUAGE)

XML_NAME := $(LANGUAGE)wiki-latest-pages-articles.xml.bz2
WIKIURL := https://dumps.wikimedia.org/$(LANGUAGE)wiki/latest/$(XML_NAME)
JSON_NAME := wiki-latest.json.gz

XML_FILE := $(WIKI_DIR)/$(XML_NAME)
JSON_FILE := $(WIKI_DIR)/$(JSON_NAME)
TOKENIZED_FILE := $(WIKI_DIR)/parsed.txt
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed.pckl

CHECKPOINT_TYPE_PATH := $(CHECKPOINT_DIR_LANG)/types_$(MAX_TRAIN_TOKENS)
CHECKPOINT_TYPE_FILE := $(CHECKPOINT_TYPE_PATH)/model.tch
CHECKPOINT_TOKEN_PATH := $(CHECKPOINT_DIR_LANG)/tokens_$(MAX_TRAIN_TOKENS)
CHECKPOINT_TOKEN_FILE := $(CHECKPOINT_TOKEN_PATH)/model.tch
DOT:= .
UNDERSCORE:= _
STRING_ALPHA = $(subst $(DOT),$(UNDERSCORE),$(ALPHA))
STRING_BETA = $(subst $(DOT),$(UNDERSCORE),$(BETA))
TWO_STAGE_INIT_TYPE_STATE_FOLDER := $(CHECKPOINT_DIR_LANG)/two_stage_init_type_$(STRING_ALPHA)_$(STRING_BETA)_$(MAX_TRAIN_TOKENS)
TWO_STAGE_INIT_TOKEN_STATE_FOLDER := $(CHECKPOINT_DIR_LANG)/two_stage_init_token_$(STRING_ALPHA)_$(STRING_BETA)_$(MAX_TRAIN_TOKENS)
GENERATOR_RESULTS_FILE := $(RESULTS_DIR_LANG)/lstm_results.csv
# training results files
TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_training_init_token.csv
TWO_STAGE_TYPE_TRAINING_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_training_init_type.csv
# evaluation results files
TWO_STAGE_INIT_TOKEN_ON_TOKEN_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_init_token_on_token.csv
TWO_STAGE_INIT_TYPE_ON_TOKEN_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_init_type_on_token.csv
TWO_STAGE_INIT_TYPE_ON_TYPE_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_init_type_on_type.csv
TWO_STAGE_INIT_TOKEN_ON_TYPE_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_init_token_on_type.csv


all: get_wiki train_generator train_two_stage eval_generator eval_two_stage

train_two_stage: run_two_stage_type_training run_two_stage_token_training
	echo "Finished training two-stage model" $(LANGUAGE)

eval_generator: $(GENERATOR_RESULTS_FILE)
	echo "Finished evaluating generator" $(LANGUAGE)

eval_two_stage: run_two_stage_token_evaluation run_two_stage_type_evaluation
	echo "Finished evaluating two-stage model" $(LANGUAGE)

train_generator: $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Finished training model" $(LANGUAGE)

get_wiki: $(PROCESSED_DATA_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

clean:
	rm $(PROCESSED_DATA_FILE)

tune_hyperparams: tune_hyperparameters
	echo "Finished hyperparameter tuning for" $(LANGUAGE)


# Eval two-stage model
run_two_stage_type_evaluation: $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	echo "Eval models on types" $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	echo "Evaluate two-stage model initialised with a type generator"
	python3 src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --two-stage-state-folder $(TWO_STAGE_INIT_TYPE_STATE_FOLDER)\
			--adaptor-results-file $(TWO_STAGE_INIT_TYPE_ON_TYPE_RESULTS_FILE) --batch-size 64 --dataset types
	echo "Evaluate two-stage model initialised with a token generator"
	python3 src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --two-stage-state-folder $(TWO_STAGE_INIT_TOKEN_STATE_FOLDER)\
			--adaptor-results-file $(TWO_STAGE_INIT_TOKEN_ON_TYPE_RESULTS_FILE) --batch-size 64 --dataset types

# Eval two-stage model
run_two_stage_token_evaluation: $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	echo "Eval models on tokens" $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	echo "Evaluate two-stage model initialised with a type generator"
	python3 src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --two-stage-state-folder $(TWO_STAGE_INIT_TYPE_STATE_FOLDER)\
			--adaptor-results-file $(TWO_STAGE_INIT_TYPE_ON_TOKEN_RESULTS_FILE) --batch-size 64 --dataset tokens
	echo "Evaluate two-stage model initialised with a token generator"
	python3 src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --two-stage-state-folder $(TWO_STAGE_INIT_TOKEN_STATE_FOLDER)\
			--adaptor-results-file $(TWO_STAGE_INIT_TOKEN_ON_TOKEN_RESULTS_FILE) --batch-size 64 --dataset tokens

# Train two-stage model initialising with types
run_two_stage_type_training: $(CHECKPOINT_TYPE_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(TWO_STAGE_INIT_TYPE_STATE_FOLDER)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_two_stage.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --adaptor-results-file $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE) \
			--alpha $(ALPHA) --beta $(BETA) --two-stage-state-folder $(TWO_STAGE_INIT_TYPE_STATE_FOLDER) --max-train-tokens $(MAX_TRAIN_TOKENS) --epochs $(EPOCHS)

# Train two-stage model initialising with tokens
run_two_stage_token_training: $(CHECKPOINT_TOKEN_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(TWO_STAGE_INIT_TOKEN_STATE_FOLDER)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_two_stage.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --adaptor-results-file $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) \
			--alpha $(ALPHA) --beta $(BETA) --two-stage-state-folder $(TWO_STAGE_INIT_TOKEN_STATE_FOLDER) --max-train-tokens $(MAX_TRAIN_TOKENS) --epochs $(EPOCHS)

# Eval language models
$(GENERATOR_RESULTS_FILE): $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Eval models" $(GENERATOR_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval_generator.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(GENERATOR_RESULTS_FILE) --batch-size 64 --dataset tokens

# Train tokens model
$(CHECKPOINT_TOKEN_FILE): $(PROCESSED_DATA_FILE)
	echo "Train tokens model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	python src/h02_learn/train_generator.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens --max-train-tokens $(MAX_TRAIN_TOKENS)

# Train types model
$(CHECKPOINT_TYPE_FILE): $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	python src/h02_learn/train_generator.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --dataset types --max-train-tokens $(MAX_TRAIN_TOKENS)

# Preprocess Data
$(PROCESSED_DATA_FILE): $(TOKENIZED_FILE)
	echo "Process data"
	python src/h01_data/process_data.py --wikipedia-tokenized-file $(TOKENIZED_FILE) --data-file $(PROCESSED_DATA_FILE)

# Tokenize wikipedia
$(TOKENIZED_FILE): $(JSON_FILE)
	echo "Tokenize data"
	python src/h01_data/tokenizer.py --wikipedia-raw-file $(JSON_FILE) --wikipedia-tokenized-file $(TOKENIZED_FILE) --dump-size 10000 --language $(LANGUAGE)

# Preprocess wikipedia to json
$(JSON_FILE): $(XML_FILE)
	echo "Parse to JSON data"
	python -m gensim.scripts.segment_wiki -i -f $(XML_FILE) -o $(JSON_FILE)

# Get wikipedia
$(XML_FILE):
	echo "Get wiki data"
	mkdir -p $(WIKI_DIR)
	wget -P $(WIKI_DIR) $(WIKIURL)

tune_hyperparameters: $(PROCESSED_DATA_FILE) $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_DIR_LANG)/hyperparam_tuning
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/tune_pitman_yor.py --results-file $(RESULTS_DIR_LANG)/hyperparam_tuning_results --no-alphas $(NO_ALPHAS) --no-betas $(NO_BETAS)\
			--two-stage-state-folder $(CHECKPOINT_DIR_LANG)/hyperparam_tuning --data-file $(PROCESSED_DATA_FILE) --max-train-tokens $(MAX_TRAIN_TOKENS) --generator-path $(CHECKPOINT_TYPE_PATH)
