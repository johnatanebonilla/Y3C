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
import random

# Tiempo base de espera entre peticiones (en segundos)
BASE_WAIT_TIME = 3  # Ajusta este valor según sea necesario

# Función para instalar los paquetes necesarios
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

    # Instalar el modelo en español de SpaCy
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])

# Asegurarse de que se instalen los requisitos
install_requirements()

# Cargar el modelo en español de SpaCy
nlp = spacy.load("es_core_news_sm")

# Función para manejar la espera exponencial
def wait_exponentially(attempt):
    wait_time = BASE_WAIT_TIME * (2 ** attempt)
    wait_time += random.uniform(0, 1)  # Añadir una variación aleatoria
    print(f"Esperando {wait_time:.2f} segundos antes del próximo intento.")
    time.sleep(wait_time)

# Función para obtener todas las URLs de videos de un canal
def get_all_video_urls_from_channel(channel_id):
    if channel_id is None:
        return []
    videos = scrapetube.get_channel(channel_id)
    video_urls = [f"https://www.youtube.com/watch?v={video['videoId']}" for video in videos]
    return video_urls

# Función para obtener el ID del video desde la URL
def obtener_id_video(url):
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1]
    return None

# Función para obtener la información del video
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

# Función para guardar la transcripción y los metadatos en un archivo
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

# Función para eliminar el contenido entre corchetes cuadrados
def remove_square_bracket_content(text):
    return re.sub(r'\[.*?\]', '', text)

# Función para tokenizar el texto usando SpaCy
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

def main():
    parser = argparse.ArgumentParser(description="Crear corpus desde canales de YouTube")
    parser.add_argument('--channels', type=str, required=True, help='Ruta al archivo Excel con los canales')
    parser.add_argument('--output', type=str, required=True, help='Directorio para los archivos de salida')
    parser.add_argument('--lang', type=str, required=True, help='Código de idioma para los subtítulos (ej. es, en)')
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

    # Leer la lista de canales desde un archivo Excel
    channels = pd.read_excel(channels_file).to_dict(orient='records')
    
    all_video_data = []

    # Obtener URLs de videos de cada canal
    for channel in channels:
        if channel['id'] is None:
            print(f"Omitiendo {channel['name']} ({channel['handle']}) - falta el ID del canal")
            continue

        try:
            video_urls = get_all_video_urls_from_channel(channel['id'])
            for url in video_urls:
                all_video_data.append((channel['name'], channel['handle'], channel['id'], url))
        except Exception as e:
            print(f"Error al procesar {channel['name']} ({channel['handle']}): {e}")

    # Guardar URLs en un archivo TSV
    with open(urls_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Channel Name', 'Channel Handle', 'Channel ID', 'URL'])
        writer.writerows(all_video_data)

    # Leer URLs desde el archivo TSV
    df = pd.read_csv(urls_file, delimiter='\t')

    # Lista para almacenar información sobre videos sin subtítulos
    no_caption_videos = []
    attempt = 0

    for index, row in df.iterrows():
        url = row['URL']
        video_id = obtener_id_video(url.strip())

        if video_id and not transcripcion_ya_descargada(captions_folder, video_id):
            success = False
            while not success:
                try:
                    # Manejo de excepción al intentar crear el objeto YouTube
                    try:
                        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                        metadata = video_info(yt)
                    except Exception as yt_error:
                        print(f"Error al acceder a los datos del video {video_id}: {yt_error}")
                        no_caption_videos.append(row)
                        success = True
                        continue

                    # Intentar obtener la transcripción
                    try:
                        transcripcion = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    except TranscriptsDisabled:
                        print(f"Los subtítulos están deshabilitados para el video {video_id}")
                        no_caption_videos.append(row)
                        success = True  # No necesita más intentos
                        continue
                    except Exception as e:
                        print(f"Error obteniendo la transcripción para el video {video_id}: {e}")
                        no_caption_videos.append(row)
                        success = True  # No necesita más intentos
                        continue
                    
                    fecha = datetime.datetime.now().strftime("%Y-%m-%d")
                    nombre_archivo = os.path.join(captions_folder, f"{fecha}_{video_id}.txt")
                    guardar_transcripcion_y_metadata(transcripcion, metadata, nombre_archivo)
                    print(f"Transcripción y metadatos guardados: {nombre_archivo}")

                    success = True  # Si todo va bien, marcar como éxito
                    attempt = 0  # Reiniciar intentos si hay éxito

                except Exception as e:
                    print(f"Error procesando el video {video_id}: {e}")
                    attempt += 1
                    if attempt > 5:
                        print(f"Demasiados intentos fallidos para {video_id}. Abandonando.")
                        no_caption_videos.append(row)
                        success = True  # Salir del ciclo y continuar con el siguiente video
                    else:
                        wait_exponentially(attempt)  # Esperar de forma exponencial antes de reintentar

            # Añadir un retraso base para evitar sobrecarga en las API
            time.sleep(BASE_WAIT_TIME)

    # Guardar la lista de videos sin subtítulos en un archivo TSV
    no_caption_df = pd.DataFrame(no_caption_videos)
    no_caption_df.to_csv(no_caption_file, sep='\t', index=False)
    print(f"Lista de videos sin subtítulos guardada en: {no_caption_file}")

    # Leer y procesar archivos de transcripción y metadatos
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

            # Eliminar marcas entre corchetes cuadrados antes de procesar
            clean_transcription = remove_square_bracket_content(record['Transcription'])
            
            # Agregar el registro a la lista de datos
            data.append(record)

    # Convertir los datos a un DataFrame
    df_corpus = pd.DataFrame(data)

    # Crear un dataset de Hugging Face
    dataset = Dataset.from_pandas(df_corpus)

    # Convertir el dataset de nuevo a DataFrame
    df_corpus = dataset.to_pandas()

    # Dividir el DataFrame en fragmentos de menos de 200 MB y guardar en parquet
    max_size_mb = 200
    df_memory_size_mb = df_corpus.memory_usage(deep=True).sum() / 1024**2

    if df_memory_size_mb < max_size_mb:
        rows_per_chunk = len(df_corpus)
    else:
        rows_per_chunk = len(df_corpus) // (max_size_mb / (df_memory_size_mb / len(df_corpus)))

    rows_per_chunk = max(1, int(rows_per_chunk))  # Asegurarse de que rows_per_chunk sea al menos 1

    for i, chunk in enumerate(range(0, len(df_corpus), rows_per_chunk)):
        chunk_df = df_corpus.iloc[chunk:chunk + rows_per_chunk]
        output_chunk_path = os.path.join(output_dir, f'corpus_data_part_{i+1}.parquet')
        chunk_df.to_parquet(output_chunk_path, index=False)
        print(f"Fragmento {i+1} guardado con éxito en {output_chunk_path}")

    # Generar estadísticas
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

    print("\nLas estadísticas se han guardado en un archivo Excel.")
    print(f"Archivo de estadísticas guardado en: {statistics_file}")

if __name__ == "__main__":
    main()
