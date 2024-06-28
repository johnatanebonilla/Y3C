# Y3C YouTube Captions Corpus Creator

This project provides a Python script to create a corpus of YouTube transcriptions. The script fetches video URLs from YouTube channels specified in an Excel file, downloads the transcriptions in the desired language, restores punctuation and capitalization, cleans unnecessary spaces, and saves the results in parquet files. It also generates an Excel file with corpus statistics.

## Requirements

- Python 3.6 or higher

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/johnatanebonilla/y3c.git
    cd y3c
    ```

2. Ensure you have `pip` updated:
    ```sh
    python -m pip install --upgrade pip
    ```

## Usage

The `create_corpus.py` script can be run from the command line with the following arguments:

- `--channels`: Path to the Excel file containing the list of channels.
- `--output`: Directory where the output files will be saved, including the parquet files and the statistics Excel file.
- `--lang`: Language code for the captions (e.g., `es` for Spanish, `en` for English, `de` for German).

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

 ## Generated Files

The script produces the following files:

1. **channels_urls.tsv**
   - **Description:** A tab-separated file containing the list of video URLs from the specified YouTube channels.
   - **Columns:**
     - Channel Name
     - Channel Handle
     - Channel ID
     - URL

2. **channels_urls_no_caption.tsv**
   - **Description:** A tab-separated file listing videos without available captions.
   - **Columns:**
     - Channel Name
     - Channel Handle
     - Channel ID
     - URL

3. **corpus/**
   - **Description:** A directory containing text files with transcriptions and metadata of each video.
   - **File Naming Convention:** `YYYY-MM-DD_video_id.txt`
     - **Example:** `2024-06-28_video123456.txt`
   - **Contents:**
     - Metadata (Title, Author, Length, Views, Video URL, Video ID, Publish Date, Tags, Description)
     - Transcription text

4. **statistics.xlsx**
   - **Description:** An Excel file summarizing various statistics of the corpus.
   - **Sheets:**
     - **Total_Statistics:** Total number of titles, total length in hours:minutes:seconds, and total token count.
     - **Titles_by_Author:** Count of titles per author.
     - **Length_by_Author:** Total length of videos per author in hours:minutes:seconds.
     - **Tokens_by_Author:** Total token count per author.

5. **corpus_data_part_X.parquet**
   - **Description:** Parquet files containing chunks of the processed corpus data. Each file is part of the complete corpus.
   - **Naming Convention:** `corpus_data_part_X.parquet`
     - **Example:** `corpus_data_part_1.parquet`
   - **Columns:**
     - Title
     - Author
     - Length
     - Views
     - Video URL
     - Video ID
     - Publish Date
     - Tags
     - Description
     - Transcription
     - Transcription_punct
     - Token_Count

## Limitations

### Punctuation and Capitalization

The punctuation and capitalization of the transcriptions are processed using automated models. While these models are highly advanced, they are not perfect and may not always accurately restore the original punctuation and capitalization. Common issues may include:

- Incorrect placement of punctuation marks.
- Missing punctuation in certain contexts.
- Incorrect capitalization of words, especially in cases of proper nouns or sentence beginnings.

Users should manually review and correct the transcriptions if high accuracy is required for specific applications.


### Contact
For more information, contact johnatanebonilla@gmail.com.
