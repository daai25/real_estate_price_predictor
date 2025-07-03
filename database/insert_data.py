import psycopg2
from property_data import properties

def insert_data():
    conn = psycopg2.connect(
        dbname="real_estate_price_predictor",
        user="prod",
        password="prod",
        host="localhost",
        port=5432
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
                area_sqm, floor, availability_date, has_balcony, description
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            prop['description']
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
