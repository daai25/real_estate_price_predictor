import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# 1. Service Account JSON Pfad
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'sigma-current-442819-k1-52bc72381234.json')

# 2. Ordner-ID (aus deinem Link)
FOLDER_ID = '1Pn1qXZ0B1MHb-KuYoHqcJBrz10Bh7PPV'

# 3. Lokales Verzeichnis für Downloads
DOWNLOAD_DIR = 'downloaded_images'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 4. Authentifizieren
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=creds)

# 5. Alle Bilddateien im Ordner abrufen (paginieren)
page_token = None
file_count = 0
while True:
    response = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false",
        spaces='drive',
        fields='nextPageToken, files(id, name)',
        pageToken=page_token
    ).execute()

    for file in response.get('files', []):
        file_id = file.get('id')
        file_name = file.get('name')
        file_count += 1

        print(f"({file_count}) Lade {file_name} herunter...")

        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(DOWNLOAD_DIR, file_name), 'wb')
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

    page_token = response.get('nextPageToken', None)
    if page_token is None:
        break

print(f"✅ Fertig! {file_count} Bilder heruntergeladen.")
