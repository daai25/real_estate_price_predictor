import psycopg2

def get_all_properties():
    try:
        conn = psycopg2.connect(
            dbname="real_estate_price_predictor",
            user="postgres",
            password="postgres",
            host="localhost",
            port=5433
        )
        cur = conn.cursor()

        cur.execute("SELECT * FROM properties;")
        rows = cur.fetchall()

        colnames = [desc[0] for desc in cur.description]

        properties = [dict(zip(colnames, row)) for row in rows]

        return properties

    except Exception as e:
        print(f"Database error: {e}")
        return []

    finally:
        if conn:
            conn.close()
