LANGUAGE := yo
DATA_DIR_BASE := ./data
WIKI_DIR := $(DATA_DIR_BASE)/wiki/$(LANGUAGE)

XML_NAME := $(LANGUAGE)wiki-latest-pages-articles.xml.bz2
WIKIURL := https://dumps.wikimedia.org/$(LANGUAGE)wiki/latest/$(XML_NAME)
JSON_NAME := wiki-latest.json.gz

XML_FILE := $(WIKI_DIR)/$(XML_NAME)
JSON_FILE := $(WIKI_DIR)/$(JSON_NAME)
TOKENIZED_FILE := $(WIKI_DIR)/parsed.txt


get_wiki: $(TOKENIZED_FILE)
	echo "Finished getting wikipedia data" $(LANGUAGE)

# Tokenize wikipedia
$(TOKENIZED_FILE): $(JSON_FILE)
	echo "Tokenize data"
	python src/tokenizer.py --wikipedia-raw-file $(JSON_FILE) --wikipedia-tokenized-file $(TOKENIZED_FILE) --dump-size 10000

# Preprocess wikipedia to json
$(JSON_FILE): $(XML_FILE)
	echo "Parse to JSON data"
	python -m gensim.scripts.segment_wiki -i -f $(XML_FILE) -o $(JSON_FILE)

# Get wikipedia
$(XML_FILE):
	echo "Get wiki data"
	mkdir -p $(WIKI_DIR)
	wget -P $(WIKI_DIR) $(WIKIURL)
