import psycopg2
from geopy.geocoders import Nominatim, Photon
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import re
import csv

# 🔷 DB-Verbindung
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="real_estate_price_predictor",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

# 🔷 Geolocator initialisieren
geolocator1 = Nominatim(user_agent="real_estate_geoloc", timeout=10)
geolocator2 = Photon(user_agent="real_estate_geoloc_fallback", timeout=10)

# 🔷 Adressen ohne Koordinaten abfragen
cur.execute("""
    SELECT id, address, zip_code, city
    FROM properties
    WHERE latitude IS NULL OR longitude IS NULL
""")
rows = cur.fetchall()
print(f"🔷 {len(rows)} Adressen ohne Koordinaten gefunden.")

fehlgeschlagen = []

def clean_address(street, zip_code, city):
    # problematische Wörter entfernen
    street = re.sub(r'(?i)\b(zone|quartier|proche|village|situation|maison|immeuble|appartement|au calme.*)\b', '', street).strip()
    return f"{street}, {zip_code} {city}, Switzerland"

def geocode_address(address):
    try:
        loc = geolocator1.geocode(address)
        if not loc:
            loc = geolocator2.geocode(address)
        return loc
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"  ⏳ Timeout/ServiceError: {e}")
        return None

for id_, street, zip_code, city in rows:
    raw_address = f"{street}, {zip_code} {city}, Switzerland"
    address = clean_address(street, zip_code, city)
    print(f"📍 Geocoding: {address}")
    location = geocode_address(address)

    if location:
        lat, lon = location.latitude, location.longitude
        print(f"    ✅ Gefunden: Lat: {lat}, Lon: {lon}")
        cur.execute(
            "UPDATE properties SET latitude=%s, longitude=%s WHERE id=%s",
            (lat, lon, id_)
        )
        conn.commit()
    else:
        print(f"    ❌ Nicht gefunden: {raw_address}")
        fehlgeschlagen.append((id_, raw_address))

    time.sleep(5)  # langsamer!

cur.close()
conn.close()

print("✅ Fertig.")

if fehlgeschlagen:
    print("\n❌ Folgende Adressen konnten nicht gefunden werden (in CSV gespeichert):")
    with open("fehlgeschlagene_adressen.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Adresse"])
        writer.writerows(fehlgeschlagen)

    for id_, addr in fehlgeschlagen:
        print(f"ID {id_}: {addr}")
else:
    print("🎉 Alle Adressen erfolgreich geokodiert.")