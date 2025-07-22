import psycopg2
import sys
import os

# Add the path to the directory containing the properties_data.py
sys.path.append(os.path.abspath("data_acquisition/data_preparation/cleaned_data"))

# Now import the properties list
from exported_properties import properties

def update_properties_with_coordinates():
    conn = psycopg2.connect(
        dbname="real_estate_price_predictor",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433
    )
    cur = conn.cursor()

    # Ensure latitude and longitude columns exist

    # Update using id from each property
    for prop in properties:
        cur.execute("""
            UPDATE properties
            SET latitude = %s,
                longitude = %s
            WHERE address = %s;
        """, (prop['latitude'], prop['longitude'], prop['address']))

    conn.commit()
    conn.close()
    print("All coordinates updated successfully.")

if __name__ == "__main__":
    update_properties_with_coordinates()