LANGUAGE := yo
ALPHA := 0.5
BETA := 1
DATA_DIR_BASE := ./data
DATA_DIR_LANG := $(DATA_DIR_BASE)/$(LANGUAGE)
WIKI_DIR := $(DATA_DIR_LANG)/wiki
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR_BASE)/$(LANGUAGE)
RESULTS_DIR_BASE := ./results
RESULTS_DIR_LANG := $(RESULTS_DIR_BASE)/$(LANGUAGE)
# these files don't exist, so that two stage training and evaluation
# are triggered each time the makefile script is run
TWO_STAGE_TOKEN_TRAINING := run_two_stage_token_training
TWO_STAGE_TYPE_TRAINING := run_two_stage_type_training
TWO_STAGE_TOKEN_EVALUATION := run_two_stage_token_evaluation
TWO_STAGE_TYPE_EVALUATION := run_two_stage_type_evaluation

XML_NAME := $(LANGUAGE)wiki-latest-pages-articles.xml.bz2
WIKIURL := https://dumps.wikimedia.org/$(LANGUAGE)wiki/latest/$(XML_NAME)
JSON_NAME := wiki-latest.json.gz

XML_FILE := $(WIKI_DIR)/$(XML_NAME)
JSON_FILE := $(WIKI_DIR)/$(JSON_NAME)
TOKENIZED_FILE := $(WIKI_DIR)/parsed.txt
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed_$(TRAIN_NUM).pckl

CHECKPOINT_TYPE_PATH := $(CHECKPOINT_DIR_LANG)/types
CHECKPOINT_TYPE_FILE := $(CHECKPOINT_TYPE_PATH)/model.tch
CHECKPOINT_TOKEN_PATH := $(CHECKPOINT_DIR_LANG)/tokens
CHECKPOINT_TOKEN_FILE := $(CHECKPOINT_TOKEN_PATH)/model.tch
DOT:= .
UNDERSCORE:= _
STRING_ALPHA = $(subst $(DOT),$(UNDERSCORE),$(ALPHA))
STRING_BETA = $(subst $(DOT),$(UNDERSCORE),$(BETA))
ADAPTOR_STATE_FILE := $(CHECKPOINT_DIR_LANG)/adaptor_state_file_$(STRING_ALPHA)_$(STRING_BETA)
GENERATOR_RESULTS_FILE := $(RESULTS_DIR_LANG)/lstm_results.csv
TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_training_results_token_init.csv
TWO_STAGE_TYPE_TRAINING_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_training_results_type_init.csv
TWO_STAGE_TOKEN_EVALUATION_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_results_token_init.csv
TWO_STAGE_TYPE_EVALUATION_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_evaluation_results_type_init.csv


all: get_wiki train two_stage eval_generator

two_stage: $(TWO_STAGE_TOKEN_TRAINING) $(TWO_STAGE_TYPE_TRAINING)
	echo "Finished training two-stage model" $(LANGUAGE)

eval_generator: $(GENERATOR_RESULTS_FILE)
	echo "Finished evaluating model" $(LANGUAGE)

eval_two_stage: $(TWO_STAGE_TOKEN_EVALUATION) $(TWO_STAGE_TYPE_EVALUATION)
	echo "Finished evaluating model" $(LANGUAGE)

train: $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Finished training model" $(LANGUAGE)

get_wiki: $(PROCESSED_DATA_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

clean:
	rm $(PROCESSED_DATA_FILE)


# Eval two-stage model
$(TWO_STAGE_TYPE_EVALUATION): $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	echo "Eval models" $(GENERATOR_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(TWO_STAGE_TYPE_EVALUATION_RESULTS_FILE) --batch-size 64 --dataset types

# Eval two-stage model
$(TWO_STAGE_TOKEN_EVALUATION): $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE)
	echo "Eval models" $(GENERATOR_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval_two_stage.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(TWO_STAGE_TOKEN_EVALUATION_RESULTS_FILE) --batch-size 64 --dataset tokens

# Eval language models
$(GENERATOR_RESULTS_FILE): $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Eval models" $(GENERATOR_RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval_generator.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(GENERATOR_RESULTS_FILE) --batch-size 64 --dataset tokens

# Train two-stage model initialising with types
$(TWO_STAGE_TYPE_TRAINING): $(CHECKPOINT_TYPE_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	mkdir -p $(CHECKPOINT_TYPE_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_pitman_yor.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --dataset tokens \
			--adaptor-results-file $(TWO_STAGE_TYPE_TRAINING_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE)

# Train two-stage model initialising with tokens
$(TWO_STAGE_TOKEN_TRAINING): $(CHECKPOINT_TOKEN_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_pitman_yor.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens \
			--adaptor-results-file $(TWO_STAGE_TOKEN_TRAINING_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE)

# Train tokens model
$(CHECKPOINT_TOKEN_FILE): $(PROCESSED_DATA_FILE)
	echo "Train tokens model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens

# Train types model
$(CHECKPOINT_TYPE_FILE): $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --dataset types

# Preprocess Data
$(PROCESSED_DATA_FILE): $(TOKENIZED_FILE)
	echo "Process data"
	python src/h01_data/process_data.py --wikipedia-tokenized-file $(TOKENIZED_FILE) --data-file $(PROCESSED_DATA_FILE) --sample-size $(TRAIN_NUM)

# Tokenize wikipedia
$(TOKENIZED_FILE): $(JSON_FILE)
	echo "Tokenize data"
	python src/h01_data/tokenizer.py --wikipedia-raw-file $(JSON_FILE) --wikipedia-tokenized-file $(TOKENIZED_FILE) --dump-size 10000

# Preprocess wikipedia to json
$(JSON_FILE): $(XML_FILE)
	echo "Parse to JSON data"
	python -m gensim.scripts.segment_wiki -i -f $(XML_FILE) -o $(JSON_FILE)

# Get wikipedia
$(XML_FILE):
	echo "Get wiki data"
	mkdir -p $(WIKI_DIR)
	wget -P $(WIKI_DIR) $(WIKIURL)
