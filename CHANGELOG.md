# CHANGELOG

All notable changes to this project will be documented in this file.

## [1.1.3] - 17-03-2023

### Added

    - optional command line arguments to specify number of top keywords and number of most similar documents to be searched for keywords. This is by default set to 5 and len(corpus) if no command line arguments are provided.
    - unittests

### Changed

    - readme contains instructions on how to run scripts
    - tfidf_search script now supports reading into corpus from pdfs and txt files only

### Fixed

    - bug in search_documents where used corpus instead of documents
    - bug in pagenumbers

## [1.1.2] - 10-03-2023

### Unreleased

    - edit readme with instructions on how run scripts

### Added

    - changelog for sematic versioning
    - script to extract pdfs
    - more comments for readability

### Changed

    - preprocessor now removes html tags as well
    - tf-idf search now outputs a csv file with extracts and page numbers and document names of the doc where the keyword is present
    - change variables names in script to reflect terminology used in team (corpus, documents, keywords, extracts)

### Fixed

    - Character encoding issues (ignore characters that are not utf-8)

## [1.1.1] - 03-03-2023

### Added

- Initial scripts for preprocessing and two types of keyword extraction + search

## [1.1.2] - 21-04-2023

### Unreleased

	- Generate text using RNN

### Added

	- Generate text using markov chain

### Changed
	
	- pdf2text now accepts either single files or folders

## [1.1.3] - 28-04-2023


### Added

	- Generate text using RNN

	### Changed
	
	- markov chain script now takes a seed phrase as input for text generation
	
## [1.1.4] - 05-05-2023


### Changed
	
	- tfidf_search now accepts notes folder instead of json
	
## [1.1.5] - 07-05-2023

### Added

Causal Language Model based text generation

## [1.1.6] - 14-05-2023

### Changed

Lm.py now accepts prompt_file instead of prompt.

### Added

Seq2Seq Language Model based text generation

## [1.1.7] - 22-05-2023

### Changed

Refactored scripts for readability. Combined lm and text2text to one script.

### Added

Sleeper Demo
Unittests for markov chain
## [1.1.8] - 27-05-2023

### Added

Added haiku.py

## [1.1.9] - 10-06-2023

###Changed

small changes to read_data module, refactoring search and updates to unittests.
added main function to each generate script to make interface consistent.

### Added

Added word_arithmetic.py
API Reference