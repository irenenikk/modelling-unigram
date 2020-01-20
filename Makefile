LANGUAGE := yo
DATA_DIR_BASE := ./data
DATA_DIR_LANG := $(DATA_DIR_BASE)/$(LANGUAGE)
WIKI_DIR := $(DATA_DIR_LANG)/wiki
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR_LANG := ./$(CHECKPOINT_DIR_BASE)/$(LANGUAGE)

XML_NAME := $(LANGUAGE)wiki-latest-pages-articles.xml.bz2
WIKIURL := https://dumps.wikimedia.org/$(LANGUAGE)wiki/latest/$(XML_NAME)
JSON_NAME := wiki-latest.json.gz

XML_FILE := $(WIKI_DIR)/$(XML_NAME)
JSON_FILE := $(WIKI_DIR)/$(JSON_NAME)
TOKENIZED_FILE := $(WIKI_DIR)/parsed.txt
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed.pckl

CHECKPOINT_PATH := $(CHECKPOINT_DIR_LANG)/baseline
CHECKPOINT_FILE := $(CHECKPOINT_PATH)/results.csv


all: get_wiki

train: $(CHECKPOINT_FILE)
	echo "Finished training model" $(LANGUAGE)

get_wiki: $(PROCESSED_DATA_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

clean:
	# rm $(TOKENIZED_FILE) $(PROCESSED_DATA_FILE)
	rm $(PROCESSED_DATA_FILE)

# Train Model
$(CHECKPOINT_FILE): $(PROCESSED_DATA_FILE)
	echo "Train model" $(CHECKPOINT_FILE)
	mkdir -p $(CHECKPOINT_PATH)
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_PATH)

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
