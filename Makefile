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
TWO_STAGE_TOKEN_TRAINING := run_two_stage_token_training
TWO_STAGE_TYPE_TRAINING := run_two_stage_type_training

XML_NAME := $(LANGUAGE)wiki-latest-pages-articles.xml.bz2
WIKIURL := https://dumps.wikimedia.org/$(LANGUAGE)wiki/latest/$(XML_NAME)
JSON_NAME := wiki-latest.json.gz

XML_FILE := $(WIKI_DIR)/$(XML_NAME)
JSON_FILE := $(WIKI_DIR)/$(JSON_NAME)
TOKENIZED_FILE := $(WIKI_DIR)/parsed.txt
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed.pckl

CHECKPOINT_TYPE_PATH := $(CHECKPOINT_DIR_LANG)/types
CHECKPOINT_TYPE_FILE := $(CHECKPOINT_TYPE_PATH)/model.tch
CHECKPOINT_TOKEN_PATH := $(CHECKPOINT_DIR_LANG)/tokens
CHECKPOINT_TOKEN_FILE := $(CHECKPOINT_TOKEN_PATH)/model.tch
DOT:= .
UNDERSCORE:= _
STRING_ALPHA = $(subst $(DOT),$(UNDERSCORE),$(ALPHA))
STRING_BETA = $(subst $(DOT),$(UNDERSCORE),$(BETA))
ADAPTOR_STATE_FILE := $(CHECKPOINT_DIR_LANG)/adaptor_state_file_$(STRING_ALPHA)_$(STRING_BETA)
RESULTS_FILE := $(RESULTS_DIR_LANG)/lstm_results.csv
ADAPTOR_TOKEN_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_results_token_init.csv
ADAPTOR_TYPE_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_results_type_init.csv


all: get_wiki train eval two_stage

two_stage: $(ADAPTOR_TYPE_RESULTS_FILE) $(ADAPTOR_TOKEN_RESULTS_FILE)
	echo "Finished training two-stage model" $(LANGUAGE)

eval: $(RESULTS_FILE)
	echo "Finished evaluating model" $(LANGUAGE)

train: $(TWO_STAGE_TOKEN_TRAINING) $(TWO_STAGE_TYPE_TRAINING)
	echo "Finished training model" $(LANGUAGE)

get_wiki: $(PROCESSED_DATA_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

clean:
	rm $(PROCESSED_DATA_FILE)

# Train two-stage model initialising with types
$(TWO_STAGE_TYPE_TRAINING): $(CHECKPOINT_TYPE_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	mkdir -p $(CHECKPOINT_TYPE_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_pitman_yor.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --dataset tokens \
			--adaptor-results-file $(ADAPTOR_TYPE_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE) --train-num $(TRAIN_NUM)

# Train two-stage model initialising with tokens
$(TWO_STAGE_TOKEN_TRAINING): $(CHECKPOINT_TOKEN_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train_pitman_yor.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens \
			--adaptor-results-file $(ADAPTOR_TOKEN_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE) --train-num $(TRAIN_NUM)

# Eval language models
$(RESULTS_FILE): $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Eval models" $(RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval_generator.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(RESULTS_FILE) --dataset tokens

# Train tokens model
$(CHECKPOINT_TOKEN_FILE): $(PROCESSED_DATA_FILE)
	echo "Train tokens model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens --train-num $(TRAIN_NUM)

# Train types model
$(CHECKPOINT_TYPE_FILE): $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --generator-path $(CHECKPOINT_TYPE_PATH) --dataset types --train-num $(TRAIN_NUM)

# Preprocess Data
$(PROCESSED_DATA_FILE): $(TOKENIZED_FILE)
	echo "Process data"
	python src/h01_data/process_data.py --wikipedia-tokenized-file $(TOKENIZED_FILE) --data-file $(PROCESSED_DATA_FILE)

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
