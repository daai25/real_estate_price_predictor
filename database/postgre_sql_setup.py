import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connection settings to default postgres DB (admin level)
admin_conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",       # assumes default admin user
    password="postgres",   # change if needed
    host="localhost",
    port=5433
)
admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
admin_cur = admin_conn.cursor()

# Step 1: Create database
try:
    admin_cur.execute("CREATE DATABASE real_estate_price_predictor;")
    print("Database created.")
except psycopg2.errors.DuplicateDatabase:
    print("Database already exists.")

# Step 2: Create user and grant privileges
try:
    admin_cur.execute("CREATE USER prod WITH PASSWORD 'prod';")
    print("User created.")
except psycopg2.errors.DuplicateObject:
    print("User already exists.")

# Grant privileges
admin_cur.execute("""
    GRANT ALL PRIVILEGES ON DATABASE real_estate_price_predictor TO prod;
""")

admin_cur.close()
admin_conn.close()

# Step 3: Connect to new DB as admin to create tables
conn = psycopg2.connect(
    dbname="real_estate_price_predictor",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)
cur = conn.cursor()

# Step 4: Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    title TEXT,
    address TEXT,
    zip_code INTEGER,
    city TEXT,
    region TEXT,
    price INTEGER,
    rooms REAL,
    area_sqm REAL,
    floor INTEGER,
    availability_date DATE,
    has_balcony BOOLEAN,
    description TEXT,
    is_rental BOOLEAN
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
cur.close()
conn.close()

print("Database and tables initialized successfully.")
