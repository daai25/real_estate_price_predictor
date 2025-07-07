import json
import os
import psycopg2

def insert_data():
    with open(os.path.abspath("data_acquisition/data_preparation/cleaned_data/cleaned_data.json"), "r", encoding="utf-8") as f:
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
                area_sqm, floor, availability_date, has_balcony, description, is_rental
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            prop['is_rental']
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

if __name__ == "__main__":
    insert_data()
