# YouTube Transcription Corpus Creator

This project provides a Python script to create a corpus of YouTube transcriptions. The script fetches video URLs from YouTube channels specified in an Excel file, downloads the transcriptions in the desired language, restores punctuation and capitalization, cleans unnecessary spaces, and saves the results in parquet files. It also generates an Excel file with corpus statistics.

## Requirements

- Python 3.6 or higher

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/your-username/repository-name.git
    cd repository-name
    ```

2. Ensure you have `pip` updated:
    ```sh
    python -m pip install --upgrade pip
    ```

## Usage

The `create_corpus.py` script can be run from the command line with the following arguments:

- `--channels`: Path to the Excel file containing the list of channels.
- `--output`: Directory where the output files will be saved, including the parquet files and the statistics Excel file.
- `--lang`: Language code for the captions (e.g., `es` for Spanish, `en` for English).

### Example

```sh
python create_corpus.py --channels channels_test.xlsx --output folder_channels --lang es
 ```

### Channel File Format
The channels Excel file (channels_test.xlsx) should have the following structure:

| name                    | handle                | id                        |
|-------------------------|-----------------------|---------------------------|
| Parlamento de Canarias  | @ParcanEs             | UCv7xnuWoLWJNEXNWIGkP19g  |
| Cabildo de La Gomera    | @cabildodelagomera6300| UC_mSsNb4Irl2KJYBz3iVEGQ  |

## Process Description

- Install requirements: The script automatically installs the necessary packages.
- Fetch video URLs: Video URLs are fetched from the specified channels.
- Download transcriptions: Transcriptions are downloaded in the specified language.
- Restore punctuation and capitalization: Punctuation and capitalization are restored in the transcriptions.
- Clean spaces: Unnecessary spaces around punctuation are cleaned.
- Save to parquet: Results are saved in parquet files, each less than 200 MB.
- Generate statistics: Statistics on the number of titles, total duration, and number of tokens are generated and saved in an Excel file.

### Contact
For more information, contact johnatanebonilla@gmail.com.
