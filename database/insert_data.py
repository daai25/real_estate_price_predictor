import json
import os
import psycopg2

def insert_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "data_acquisition", "data_preparation", "cleaned_data", "cleaned_data.json")


    with open(file_path, "r", encoding="utf-8") as f:
        properties = json.load(f)

        conn = psycopg2.connect(
            dbname="real_estate_price_predictor",
            user="postgres",
            password="postgres",
            host="localhost",
            port=5433
        )
        cur = conn.cursor()

        for prop in properties:
            # Convert area_sqm to float if it's a string or None
            try:
                area = float(prop['area_sqm']) if prop['area_sqm'] else None
            except ValueError:
                area = None

            try:
                rooms = float(prop['rooms']) if prop['rooms'] else None
            except ValueError:
                rooms = None

            # Insert into properties
            cur.execute("""
                INSERT INTO properties (
                    title, address, zip_code, city, region, price, rooms,
                    area_sqm, floor, availability_date, has_balcony, description, is_rental, is_new, has_view, has_garden, has_parking, has_air_conditioning
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                prop['title'],
                prop['address'],
                prop['zip_code'],
                prop['city'],
                prop['region'],
                prop['price'],
                rooms,
                area,
                prop['floor'],
                prop['availability_date'],
                prop['has_balcony'],
                prop['description'],
                prop['is_rental'],
                is_new(prop['description']),
                has_view(prop['description']),
                has_garden(prop['description']),
                has_parking(prop['description']),
                has_air_conditioning(prop['description'] )
            ))

            property_id = cur.fetchone()[0]

            # Insert images
            for url in prop.get('image_urls', []):
                cur.execute("""
                    INSERT INTO images (property_id, url)
                    VALUES (%s, %s);
                """, (property_id, url))

        conn.commit()
        conn.close()
        print("All data inserted successfully.")


def hash_description(description: str):
    hash = set(description.lower().rsplit())
    return hash


def is_new(description):

    # Having repeated values here will not affect memory usage
    new_synomys = {
        # French
        'neuf', 'neuve', 'neufs', 'neuves', 'neufve', 'neufves', 'neuf', 'nouveau', 'nouvelle', 'rénové', 'rénovée', 'rénovées', 'rénovation', 'moderne', 'restauré', 'restaurée', 'restaurées', 'restauration', 'refait', 'refaite', 'refaites', 'refait à neuf', 'neuf', 'neuve', 'neuves', 'neufs',
        # German
        'neu', 'neues', 'neuer', 'neuem', 'neuen', 'modern', 'modernisiert', 'renoviert', 'saniert', 'erneuert', 'restauriert', 'neubau', 'neubauten',
        # English
        'new', 'modern', 'renovated', 'renewed', 'refurbished', 'restored', 'recently renovated', 'brand new', 'fully renovated', 'newly built', 'new build',
        # Italian
        'nuovo', 'nuova', 'nuovi', 'nuove', 'moderno', 'moderna', 'moderni', 'moderne', 'rinnovato', 'rinnovata', 'rinnovati', 'rinnovate', 'ristrutturato', 'ristrutturata', 'ristrutturati', 'ristrutturate', 'restaurato', 'restaurata', 'restaurati', 'restaurate'
    }
    
    hash = hash_description(description)

    return bool(hash.intersection(new_synomys)) # if the intersection is not empty then it will be true

def has_view(description):

    # Having repeated values here will not affect memory usage
    view_synomys = {
        # French
        'vue', 'panoramique', 'imprenable', 'dégagée', 'sur lac', 'sur montagne', 'sur jardin', 'sur parc', 'vue mer', 'vue montagne', 'vue lac',
        # German
        'aussicht', 'blick', 'panoramablick', 'seeblick', 'bergsicht', 'gartenblick', 'parkblick', 'meerblick',
        # English
        'view', 'panoramic', 'unobstructed',
        # Italian
        'vista', 'panoramica', 'vista lago',
    }
    hash = hash_description(description)

    return bool(hash.intersection(view_synomys)) # if the intersection is not empty then it will be true


def has_garden(description):
    garden_synomys = {
        # French
        'jardin', 'parc', 'pelouse', 'verger',
        # German
        'garten', 'park', 'rasen', 'obstgarten',
        # English
        'garden', 'park', 'lawn', 'orchard',
        # Italian
        'giardino', 'parco', 'prato', 'frutteto'
    }
    hash = hash_description(description)
    return bool(hash.intersection(garden_synomys))

def has_parking(description):
    parking_synomys = {
        # French
        'parking', 'garage', 'stationnement', 'place de parc', 'box',
        # German
        'parkplatz', 'garage', 'stellplatz', 'carport',
        # English
        'parking', 'garage', 'parking space', 'carport',
        # Italian
        'parcheggio', 'garage', 'posto auto', 'box'
    }
    hash = hash_description(description)
    return bool(hash.intersection(parking_synomys))

def has_air_conditioning(description):
    ac_synomys = {
        # French
        'climatisation', 'climatiseur', 'air conditionné',
        # German
        'klimaanlage', 'klimagerät', 'air conditioning',
        # English
        'air conditioning', 'ac', 'air conditioner',
        # Italian
        'aria condizionata', 'condizionatore', 'climatizzazione'
    }
    hash = hash_description(description)
    return bool(hash.intersection(ac_synomys))

if __name__ == "__main__":
    insert_data()
