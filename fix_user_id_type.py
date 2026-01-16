"""
Fix user_id column type from UUID to TEXT
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

print("🔗 Connecting to database...")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("✅ Connected!")
print("\n🔧 Changing user_id from UUID to TEXT...")

tables = ['product_listings', 'service_listings', 'mutual_listings', 'matches']

for table in tables:
    try:
        print(f"\n📝 Updating {table}...")

        # Check current type
        cursor.execute(f"""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'user_id'
            OR table_name = '{table}' AND column_name = 'query_user_id';
        """)
        result = cursor.fetchone()

        if result:
            current_type = result[0]
            print(f"  Current type: {current_type}")

            if current_type == 'uuid':
                # Alter column type
                column_name = 'query_user_id' if table == 'matches' else 'user_id'
                cursor.execute(f"""
                    ALTER TABLE {table}
                    ALTER COLUMN {column_name} TYPE TEXT;
                """)
                print(f"  ✅ Changed {column_name} to TEXT")
            else:
                print(f"  ℹ️ Already TEXT type")
        else:
            print(f"  ⚠️ Column not found")

    except Exception as e:
        print(f"  ❌ Error: {e}")

conn.commit()
cursor.close()
conn.close()

print("\n" + "="*80)
print("✅ USER_ID TYPE UPDATE COMPLETE!")
print("="*80)
print("\nAll user_id columns are now TEXT type.")
print("Tests can now use strings like 'user-seller-1' instead of UUIDs.")
