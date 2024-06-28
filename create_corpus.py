import subprocess
import sys

# Instalar los requisitos necesarios
def install_requirements():
    required_packages = [
        "scrapetube",
        "youtube-transcript-api",
        "pytube",
        "pandas",
        "transformers",
        "deepmultilingualpunctuation",
        "nltk",
        "openpyxl",
    ]
    
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

import argparse
import os
import pandas as pd
import datetime
import time
import glob
import re
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
from deepmultilingualpunctuation import PunctuationModel
import nltk

nltk.download('punkt')

# Función para obtener URLs de videos de un canal
def get_all_video_urls_from_channel(channel_id):
    if channel_id is None:
        return []
    videos = scrapetube.get_channel(channel_id)
    video_urls = [f"https://www.youtube.com/watch?v={video['videoId']}" for video in videos]
    return video_urls

# Función para obtener ID de video desde URL
def obtener_id_video(url):
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1]
    return None

# Función para obtener información del video
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

# Función para guardar transcripción y metadata en archivo
def guardar_transcripcion_y_metadata(transcripcion, metadata, nombre_archivo):
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
        for key, value in metadata.items():
            archivo.write(f"{key}: {value}\n")
        archivo.write("\nTranscription:\n")
        for item in transcripcion:
            start_time = str(datetime.timedelta(seconds=int(item['start'])))
            archivo.write(f"[{start_time}] {item['text']}\n")

# Función para verificar si la transcripción ya ha sido descargada
def transcripcion_ya_descargada(output_folder, video_id):
    archivos_existentes = os.path.join(output_folder, f"*_{video_id}.txt")
    return len(glob.glob(archivos_existentes)) > 0

# Función para dividir el texto en fragmentos
def split_text_into_chunks(text, tokenizer, max_length):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(tokenizer.tokenize(word))
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Función para restaurar la puntuación en fragmentos de texto
def restore_punctuation_in_chunks(text, model, chunk_size=256):
    punctuated_text = ""
    chunks = split_text_into_chunks(text, tokenizer, chunk_size)
    for chunk in chunks:
        try:
            punctuated_chunk = model.restore_punctuation(chunk)
            punctuated_text += punctuated_chunk + " "
        except Exception as e:
            print(f"Error processing chunk: {e}")
            punctuated_text += chunk
    return punctuated_text.strip()

# Función para capitalizar texto
def get_result_text_capitalization(list_entity, text):
    result_words = []
    tmp_word = ""

    for idx, entity in enumerate(list_entity):
        tag = entity["entity"]
        word = entity["word"]
        start = entity["start"]
        end = entity["end"]

        subword = False
        if word[0] == "#":
            subword = True
            if tmp_word == "":
                p_s = list_entity[idx-1]["start"]
                p_e = list_entity[idx-1]["end"]
                tmp_word = text[p_s:p_e] + text[start:end]
            else:
                tmp_word = tmp_word + text[start:end]
            word = tmp_word
        else:
            tmp_word = ""
            word = text[start:end]

        if tag == "l":
            word = word
        elif tag == "u":
            word = word.capitalize()

        if subword:
            result_words[-1] = word
        else:
            result_words.append(word)

    return " ".join(result_words)

# Función para limpiar espacios innecesarios alrededor de la puntuación
def clean_punctuation(text):
    if isinstance(text, str):
        text = re.sub(r'\s+([,.;:?!])', r'\1', text)
        text = re.sub(r'([¿¡])\s+', r'\1', text)
    return text

# Inicializar el modelo y tokenizer
punctuation_model = PunctuationModel(model="kredor/punctuate-all")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
capitalization_model_path = "VOCALINLP/spanish_capitalization_punctuation_restoration_sanivert"
capitalization_model = AutoModelForTokenClassification.from_pretrained(capitalization_model_path)
capitalization_tokenizer = AutoTokenizer.from_pretrained(capitalization_model_path)
capitalization_pipe = pipeline("token-classification", model=capitalization_model, tokenizer=capitalization_tokenizer, device=0 if torch.cuda.is_available() else -1)

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

    # Leer lista de canales desde archivo Excel
    channels = pd.read_excel(channels_file).to_dict(orient='records')
    
    all_video_data = []

    # Obtener URLs de videos de cada canal
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

    # Guardar URLs en archivo TSV
    with open(urls_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Channel Name', 'Channel Handle', 'Channel ID', 'URL'])
        writer.writerows(all_video_data)

    # Leer URLs desde archivo TSV
    df = pd.read_csv(urls_file, delimiter='\t')

    # Lista para almacenar información sobre videos sin subtítulos
    no_caption_videos = []

    for index, row in df.iterrows():
        url = row['URL']
        video_id = obtener_id_video(url.strip())
        if video_id and not transcripcion_ya_descargada(captions_folder, video_id):
            try:
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                metadata = video_info(yt)
                # Intentar obtener la transcripción en el idioma especificado
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
                # Agregar un retraso para evitar límites de tasa
                time.sleep(3)

    # Guardar la lista de videos sin subtítulos en archivo TSV
    no_caption_df = pd.DataFrame(no_caption_videos)
    no_caption_df.to_csv(no_caption_file, sep='\t', index=False)
    print(f"List of videos without captions saved to: {no_caption_file}")

    # Leer y procesar los archivos de transcripción y metadatos
    data = []

    for filename in os.listdir(captions_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(captions_folder, filename)

            # Leer el contenido del archivo
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # Dividir el contenido en líneas
            lines = content.split('\n')

            # Extraer los campos relevantes
            record = {}
            record['Title'] = lines[0].split(': ')[1]
            record['Author'] = lines[1].split(': ')[1]
            record['Length'] = int(lines[2].split(': ')[1].split()[0])  # en segundos
            record['Views'] = lines[3].split(': ')[1]
            record['Video URL'] = lines[4].split(': ')[1]
            record['Video ID'] = lines[5].split(': ')[1]
            record['Publish Date'] = lines[6].split(': ')[1]
            record['Tags'] = lines[7].split(': ')[1] if len(lines[7].split(': ')) > 1 else None
            record['Description'] = lines[8].split(': ')[1] if len(lines[8].split(': ')) > 1 else None

            # Unir el texto de la transcripción
            transcription_index = lines.index('Transcription:') + 1
            transcription = ' '.join(lines[transcription_index:])
            record['Transcription'] = transcription.strip()

            # Restaurar la puntuación de la transcripción
            record['Transcription_punct'] = restore_punctuation_in_chunks(record['Transcription'], punctuation_model)

            # Agregar el registro a la lista de datos
            data.append(record)

    # Convertir los datos a un DataFrame
    df_corpus = pd.DataFrame(data)

    # Aplicar capitalización a las transcripciones
    for index, row in df_corpus.iterrows():
        chunks = split_text_into_chunks(row['Transcription_punct'], capitalization_tokenizer, max_length=512)
        capitalized_chunks = [get_result_text_capitalization(capitalization_pipe(chunk), chunk) for chunk in chunks]
        df_corpus.at[index, 'Transcription_punct'] = " ".join(capitalized_chunks)

    # Limpiar espacios innecesarios alrededor de la puntuación
    df_corpus['Transcription_punct'] = df_corpus['Transcription_punct'].apply(clean_punctuation)

    # Dividir el DataFrame en fragmentos de menos de 200 MB y guardar en parquet
    max_size_mb = 200
    rows_per_chunk = len(df_corpus) // (max_size_mb / (df_corpus.memory_usage(deep=True).sum() / len(df_corpus) / 1024**2))

    for i, chunk in enumerate(range(0, len(df_corpus), int(rows_per_chunk))):
        chunk_df = df_corpus.iloc[chunk:chunk + int(rows_per_chunk)]
        output_chunk_path = os.path.join(output_dir, f'corpus_data_part_{i+1}.parquet')
        chunk_df.to_parquet(output_chunk_path, index=False)
        print(f"Chunk {i+1} successfully saved to {output_chunk_path}")

    # Generar estadísticas
    df_corpus['Token_Count'] = df_corpus['Transcription_punct'].apply(lambda x: len(nltk.word_tokenize(x)))
    total_titles = df_corpus['Title'].count()
    total_length_seconds = df_corpus['Length'].sum()
    total_length_hms = str(datetime.timedelta(seconds=total_length_seconds))
    total_tokens = df_corpus['Token_Count'].sum()

    titles_by_author = df_corpus.groupby('Author')['Title'].count()
    length_by_author_seconds = df_corpus.groupby('Author')['Length'].sum()
    length_by_author_hms = length_by_author_seconds.apply(lambda x: str(datetime.timedelta(seconds=x)))
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
