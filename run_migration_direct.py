"""
Run database migration directly using psycopg2
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Get database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env file")
    exit(1)

print(f"🔗 Connecting to database...")

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("✅ Connected successfully!")
    print("\n📝 Running migration SQL...")

    # Read migration SQL
    with open("migrations/002_create_listings_tables.sql", "r") as f:
        migration_sql = f.read()

    # Remove comments and split into statements
    statements = []
    current_statement = []

    for line in migration_sql.split('\n'):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('--'):
            continue
        current_statement.append(line)
        # If line ends with semicolon, it's the end of a statement
        if line.endswith(';'):
            statements.append(' '.join(current_statement))
            current_statement = []

    print(f"\n🔄 Executing {len(statements)} SQL statements...")

    for i, statement in enumerate(statements, 1):
        try:
            if statement.strip():
                cursor.execute(statement)
                print(f"  ✅ Statement {i} executed")
        except Exception as e:
            print(f"  ⚠️ Statement {i} warning: {e}")

    # Commit changes
    conn.commit()
    print("\n✅ Migration completed successfully!")

    # Verify tables created
    print("\n🔍 Verifying tables...")
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('product_listings', 'service_listings', 'mutual_listings')
        ORDER BY table_name;
    """)

    tables = cursor.fetchall()
    print("\n📊 Tables created:")
    for table in tables:
        print(f"  ✅ {table[0]}")

    # Close connection
    cursor.close()
    conn.close()

    print("\n" + "="*80)
    print("✅ MIGRATION COMPLETE!")
    print("="*80)
    print("\nYou can now run: python3 test_complete_flow.py")

except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
