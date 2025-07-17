import psycopg2
import requests
import time
from tqdm import tqdm
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# === CONFIG ===
DB_PARAMS = {
    "dbname": "real_estate_price_predictor",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5433
}
TABLE = "images"
LOCAL_TMP_DIR = "tmp_images"
DRIVE_FOLDER_ID = "1Pn1qXZ0B1MHb-KuYoHqcJBrz10Bh7PPV"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, "client_secret.json")
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.json")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# === SETUP ===
if not os.path.exists(LOCAL_TMP_DIR):
    os.makedirs(LOCAL_TMP_DIR)

# === GOOGLE DRIVE AUTH ===
creds = None
if os.path.exists(TOKEN_PATH):
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_PATH, "w") as token:
        token.write(creds.to_json())

drive_service = build("drive", "v3", credentials=creds)

# === DB: fetch urls ===
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()
cur.execute(f"SELECT id, property_id, url FROM {TABLE}")
rows = cur.fetchall()

print(f"🔷 {len(rows)} Datensätze geladen. Starte Upload …")

def upload_to_drive(local_path, file_name):
    file_metadata = {
        "name": file_name,
        "parents": [DRIVE_FOLDER_ID]
    }
    media = MediaFileUpload(local_path, resumable=False)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return f"https://drive.google.com/uc?id={file.get('id')}"

def process_record(id, property_id, url):
    if url.startswith("https://drive.google.com/"):
        return

    try:
        ext = os.path.splitext(url.split("?")[0])[1]
        file_name = f"{property_id}-{id}{ext}"
        local_path = os.path.join(LOCAL_TMP_DIR, file_name)

        r = requests.get(url, headers=HEADERS, timeout=10, verify=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

        drive_url = upload_to_drive(local_path, file_name)

        cur.execute(f"UPDATE {TABLE} SET url=%s WHERE id=%s", (drive_url, id))
        conn.commit()

        os.remove(local_path)
        time.sleep(0.2)

    except Exception as e:
        print(f"❌ Fehler bei ID {id} (URL: {url}): {e}")

for row in tqdm(rows):
    process_record(*row)

cur.close()
conn.close()
print("✅ Alle Bilder verarbeitet.")