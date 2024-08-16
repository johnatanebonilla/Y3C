import subprocess
import sys
import argparse
import scrapetube
import os
import pandas as pd
import datetime
import time
import glob
import re
import csv
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import spacy
from datasets import Dataset

# Function to install required packages
def install_requirements():
    required_packages = [
        "scrapetube",
        "youtube-transcript-api",
        "pytube",
        "pandas",
        "spacy",
        "openpyxl",
        "datasets"
    ]
    
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Install SpaCy's Spanish model
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])

install_requirements()

# Load SpaCy Spanish model
nlp = spacy.load("es_core_news_sm")

# Function to get all video URLs from a channel
def get_all_video_urls_from_channel(channel_id):
    if channel_id is None:
        return []
    videos = scrapetube.get_channel(channel_id)
    video_urls = [f"https://www.youtube.com/watch?v={video['videoId']}" for video in videos]
    return video_urls

# Function to obtain video ID from URL
def obtener_id_video(url):
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1]
    return None

# Function to get video information
def video_info(yt):
    return {
        "Title": yt.title,
        "Author": yt.author,
        "Length": yt.length,
        "Views": yt.views,
        "Video URL": yt.watch_url,
        "Video ID": yt.video_id,
        "Publish Date": yt.publish_date.strftime('%Y-%m-%d') if yt.publish_date else "Not Available",
        "Tags": ', '.join(yt.keywords),
        "Description": yt.description
    }

# Function to save transcription and metadata to file
def guardar_transcripcion_y_metadata(transcripcion, metadata, nombre_archivo):
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
        for key, value in metadata.items():
            archivo.write(f"{key}: {value}\n")
        archivo.write("\nTranscription:\n")
        for item in transcripcion:
            start_time = str(datetime.timedelta(seconds=int(item['start'])))
            archivo.write(f"[{start_time}] {item['text']}\n")

# Function to check if the transcription has already been downloaded
def transcripcion_ya_descargada(output_folder, video_id):
    archivos_existentes = os.path.join(output_folder, f"*_{video_id}.txt")
    return len(glob.glob(archivos_existentes)) > 0

# Function to remove square bracket content
def remove_square_bracket_content(text):
    return re.sub(r'\[.*?\]', '', text)

# Function to tokenize text using SpaCy
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

def main():
    parser = argparse.ArgumentParser(description="Create corpus from YouTube channels")
    parser.add_argument('--channels', type=str, required=True, help='Path to the channels Excel file')
    parser.add_argument('--output', type=str, required=True, help='Directory for output files')
    parser.add_argument('--lang', type=str, required=True, help='Language code for captions (e.g., es, en)')
    args = parser.parse_args()

    channels_file = args.channels
    output_dir = args.output
    lang = args.lang
    urls_file = os.path.join(output_dir, 'channels_urls.tsv')
    captions_folder = os.path.join(output_dir, 'corpus')
    no_caption_file = os.path.join(output_dir, 'channels_urls_no_caption.tsv')
    statistics_file = os.path.join(output_dir, 'statistics.xlsx')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(captions_folder, exist_ok=True)

    # Read list of channels from Excel file
    channels = pd.read_excel(channels_file).to_dict(orient='records')
    
    all_video_data = []

    # Get URLs of videos from each channel
    for channel in channels:
        if channel['id'] is None:
            print(f"Skipping {channel['name']} ({channel['handle']}) - missing channel ID")
            continue

        try:
            video_urls = get_all_video_urls_from_channel(channel['id'])
            for url in video_urls:
                all_video_data.append((channel['name'], channel['handle'], channel['id'], url))
        except Exception as e:
            print(f"Failed to process {channel['name']} ({channel['handle']}): {e}")

    # Save URLs to TSV file
    with open(urls_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Channel Name', 'Channel Handle', 'Channel ID', 'URL'])
        writer.writerows(all_video_data)

    # Read URLs from TSV file
    df = pd.read_csv(urls_file, delimiter='\t')

    # List to store information about videos without subtitles
    no_caption_videos = []

    for index, row in df.iterrows():
        url = row['URL']
        video_id = obtener_id_video(url.strip())
        if video_id and not transcripcion_ya_descargada(captions_folder, video_id):
            try:
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                metadata = video_info(yt)
                # Try to get the transcription in the specified language
                try:
                    transcripcion = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                except TranscriptsDisabled:
                    print(f"Transcripts are disabled for video {video_id}")
                    no_caption_videos.append(row)
                    continue
                except Exception as e:
                    print(f"Error fetching transcript for video {video_id}: {e}")
                    no_caption_videos.append(row)
                    continue
                fecha = datetime.datetime.now().strftime("%Y-%m-%d")
                nombre_archivo = os.path.join(captions_folder, f"{fecha}_{video_id}.txt")
                guardar_transcripcion_y_metadata(transcripcion, metadata, nombre_archivo)
                print(f"Transcription and metadata saved: {nombre_archivo}")
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                no_caption_videos.append(row)
            finally:
                # Add a delay to avoid rate limits
                time.sleep(3)

    # Save the list of videos without captions to a TSV file
    no_caption_df = pd.DataFrame(no_caption_videos)
    no_caption_df.to_csv(no_caption_file, sep='\t', index=False)
    print(f"List of videos without captions saved to: {no_caption_file}")

    # Read and process transcription and metadata files
    data = []

    for filename in os.listdir(captions_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(captions_folder, filename)

            # Read the content of the file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split the content into lines
            lines = content.split('\n')

            # Extract relevant fields
            record = {}
            record['Title'] = lines[0].split(': ')[1]
            record['Author'] = lines[1].split(': ')[1]
            record['Length'] = int(lines[2].split(': ')[1].split()[0])  # in seconds
            record['Views'] = lines[3].split(': ')[1]
            record['Video URL'] = lines[4].split(': ')[1]
            record['Video ID'] = lines[5].split(': ')[1]
            record['Publish Date'] = lines[6].split(': ')[1]
            record['Tags'] = lines[7].split(': ')[1] if len(lines[7].split(': ')) > 1 else None
            record['Description'] = lines[8].split(': ')[1] if len(lines[8].split(': ')) > 1 else None

            # Join the transcription text
            transcription_index = lines.index('Transcription:') + 1
            transcription = ' '.join(lines[transcription_index:])
            record['Transcription'] = transcription.strip()

            # Remove square bracket content before processing
            clean_transcription = remove_square_bracket_content(record['Transcription'])
            
            # Add the record to the data list
            data.append(record)

    # Convert the data to a DataFrame
    df_corpus = pd.DataFrame(data)

    # Create a Hugging Face dataset
    dataset = Dataset.from_pandas(df_corpus)

    # Convert the dataset back to a DataFrame
    df_corpus = dataset.to_pandas()

    # Split the DataFrame into chunks of less than 200 MB and save to parquet
    max_size_mb = 200
    df_memory_size_mb = df_corpus.memory_usage(deep(True)).sum() / 1024**2

    if df_memory_size_mb < max_size_mb:
        rows_per_chunk = len(df_corpus)
    else:
        rows_per_chunk = len(df_corpus) // (max_size_mb / (df_memory_size_mb / len(df_corpus)))

    rows_per_chunk = max(1, int(rows_per_chunk))  # Ensure rows_per_chunk is at least 1

    for i, chunk in enumerate(range(0, len(df_corpus), rows_per_chunk)):
        chunk_df = df_corpus.iloc[chunk:chunk + rows_per_chunk]
        output_chunk_path = os.path.join(output_dir, f'corpus_data_part_{i+1}.parquet')
        chunk_df.to_parquet(output_chunk_path, index=False)
        print(f"Chunk {i+1} successfully saved to {output_chunk_path}")

    # Generate statistics
    df_corpus['Token_Count'] = df_corpus['Transcription'].apply(lambda x: len(tokenize_text(x)))
    total_titles = df_corpus['Title'].count()
    total_length_seconds = df_corpus['Length'].sum()
    total_length_hms = str(datetime.timedelta(seconds=int(total_length_seconds)))
    total_tokens = df_corpus['Token_Count'].sum()

    titles_by_author = df_corpus.groupby('Author')['Title'].count()
    length_by_author_seconds = df_corpus.groupby('Author')['Length'].sum().astype(int)
    length_by_author_hms = length_by_author_seconds.apply(lambda x: str(datetime.timedelta(seconds=int(x))))
    tokens_by_author = df_corpus.groupby('Author')['Token_Count'].sum()

    statistics = {
        "total_titles": [total_titles],
        "total_length_hms": [total_length_hms],
        "total_tokens": [total_tokens],
    }

    titles_by_author_df = titles_by_author.reset_index().rename(columns={'Title': 'count'})
    length_by_author_hms_df = length_by_author_hms.reset_index().rename(columns={'Length': 'length_hms'})
    tokens_by_author_df = tokens_by_author.reset_index().rename(columns={'Token_Count': 'tokens'})

    with pd.ExcelWriter(statistics_file, engine='openpyxl') as writer:
        pd.DataFrame(statistics).to_excel(writer, sheet_name='Total_Statistics', index=False)
        titles_by_author_df.to_excel(writer, sheet_name='Titles_by_Author', index=False)
        length_by_author_hms_df.to_excel(writer, sheet_name='Length_by_Author', index=False)
        tokens_by_author_df.to_excel(writer, sheet_name='Tokens_by_Author', index=False)

    print("\nStatistics have been saved to a single Excel file.")
    print(f"Statistics file saved to: {statistics_file}")

if __name__ == "__main__":
    main()
