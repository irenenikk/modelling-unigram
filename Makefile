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
DASH:= -
STRING_ALPHA = $(subst $(DOT),$(DASH),$(ALPHA))
STRING_BETA = $(subst $(DOT),$(DASH),$(BETA))
ADAPTOR_STATE_FILE := $(CHECKPOINT_DIR_LANG)/adaptor_state_file_$(STRING_ALPHA)_$(STRING_BETA)
RESULTS_FILE := $(RESULTS_DIR_LANG)/results.csv
ADAPTOR_RESULTS_FILE := $(RESULTS_DIR_LANG)/adaptor_results.csv


all: get_wiki

eval: $(RESULTS_FILE)
	echo "Finished evaluating model" $(LANGUAGE)

train: $(CHECKPOINT_TOKEN_FILE)
	echo "Finished training model" $(LANGUAGE)

get_wiki: $(PROCESSED_DATA_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

clean:
	# rm $(TOKENIZED_FILE) $(PROCESSED_DATA_FILE)
	rm $(PROCESSED_DATA_FILE)

# Eval language models
$(RESULTS_FILE): $(CHECKPOINT_TOKEN_FILE) $(CHECKPOINT_TYPE_FILE)
	echo "Eval models" $(RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval.py --data-file $(PROCESSED_DATA_FILE) --eval-path $(CHECKPOINT_DIR_LANG) --results-file $(RESULTS_FILE) --dataset tokens

# Train tokens model
$(CHECKPOINT_TOKEN_FILE): $(PROCESSED_DATA_FILE)
	echo "Train tokens model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens

# Train types model
$(CHECKPOINT_TYPE_FILE): $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_TYPE_PATH) --dataset types

# Train two-stage model initialising with types
$(CHECKPOINT_TYPE_FILE): $(PROCESSED_DATA_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TYPE_FILE)
	mkdir -p $(CHECKPOINT_TYPE_PATH)
	mkdir -p $(CHECKPOINT_TYPE_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_TYPE_PATH) --dataset tokens \
			--adaptor-results-file $(ADAPTOR_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE)

# Train two-stage model initialising with tokens
$(CHECKPOINT_TOKEN_FILE): $(PROCESSED_DATA_FILE)
	echo "Train two-stage model" $(CHECKPOINT_TOKEN_FILE)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)
	mkdir -p $(CHECKPOINT_TOKEN_PATH)_retrained
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_TOKEN_PATH) --dataset tokens \
			--adaptor-results-file $(ADAPTOR_RESULTS_FILE) --alpha $(ALPHA) --beta $(BETA) --adaptor-state-file $(ADAPTOR_STATE_FILE)

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
