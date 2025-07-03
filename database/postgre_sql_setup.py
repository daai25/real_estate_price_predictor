import psycopg2

conn = psycopg2.connect(
    dbname="real_estate_price_predictor",
    user="prod",
    password="prod",
    host="localhost",
    port=5432
)

cur = conn.cursor()

# Create the tables
cur.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    title TEXT,
    address TEXT,
    zip_code TEXT,
    city TEXT,
    region TEXT,
    price INTEGER,
    rooms REAL,
    area_sqm REAL,
    floor TEXT,
    availability_date TEXT,
    has_balcony BOOLEAN,
    description TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    property_id INTEGER REFERENCES properties(id) ON DELETE CASCADE,
    url TEXT
);
""")

conn.commit()
print("Tables created successfully.")

# (Optional) Verify connection
cur.execute("SELECT version();")
print(cur.fetchone())

conn.close()
