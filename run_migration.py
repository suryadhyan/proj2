"""
Database Migration Script

Runs the SQL migration to create matches table and update listings tables.
"""

import os
from supabase import create_client
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    exit(1)

print("🔧 Initializing Supabase client...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Read migration SQL
migration_file = Path(__file__).parent / "migrations" / "001_create_matches_table.sql"

print(f"📄 Reading migration file: {migration_file}")
with open(migration_file, 'r') as f:
    migration_sql = f.read()

print("\n" + "="*80)
print("RUNNING DATABASE MIGRATION")
print("="*80)

print("\n📋 Migration steps:")
print("  1. Create matches table")
print("  2. Add user_id and match_id to product_listings")
print("  3. Add user_id and match_id to service_listings")
print("  4. Add user_id and match_id to mutual_listings")
print("  5. Create indexes")

print("\n⚠️  WARNING: This will modify your database schema!")
response = input("\nProceed with migration? (yes/no): ")

if response.lower() != 'yes':
    print("❌ Migration cancelled.")
    exit(0)

try:
    print("\n🚀 Running migration...")

    # Split migration into statements and execute
    # Note: Supabase Python client doesn't support raw SQL execution directly
    # We need to use the REST API or run this via SQL editor

    print("\n" + "="*80)
    print("⚠️  MIGRATION INSTRUCTIONS")
    print("="*80)
    print("\nThe Supabase Python client doesn't support raw SQL execution.")
    print("Please follow these steps to run the migration:\n")

    print("OPTION 1: Use Supabase Dashboard (Recommended)")
    print("-" * 80)
    print("1. Go to: https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Click 'SQL Editor' in left sidebar")
    print("4. Click 'New Query'")
    print("5. Copy the SQL from: migrations/001_create_matches_table.sql")
    print("6. Paste into the editor")
    print("7. Click 'Run' button\n")

    print("OPTION 2: Use psql Command Line")
    print("-" * 80)
    print("1. Extract connection string from Supabase dashboard")
    print("2. Run: psql <connection-string> -f migrations/001_create_matches_table.sql\n")

    print("OPTION 3: Copy SQL Below")
    print("-" * 80)
    print("\nSQL to execute:\n")
    print(migration_sql)
    print("\n" + "="*80)

    # Verify tables exist (read-only check)
    print("\n✅ Migration file is ready at: migrations/001_create_matches_table.sql")
    print("   Please run it using one of the options above.")

except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
