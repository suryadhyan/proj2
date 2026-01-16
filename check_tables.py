"""
Quick script to check existing Supabase tables
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

print(f"🔗 Connecting to Supabase: {SUPABASE_URL}")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Try to query the information schema
try:
    # Check what tables exist
    response = supabase.rpc('exec_sql', {
        'query': """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """
    }).execute()

    print("\n📊 Existing tables in database:")
    for row in response.data:
        print(f"  - {row}")

except Exception as e:
    print(f"\n⚠️ Could not query tables via RPC: {e}")
    print("\nTrying to list tables directly...")

    # Try checking each table individually
    tables_to_check = [
        'listings',
        'product_listings',
        'service_listings',
        'mutual_listings',
        'matches'
    ]

    print("\n📋 Checking for specific tables:")
    for table in tables_to_check:
        try:
            result = supabase.table(table).select("*").limit(1).execute()
            print(f"  ✅ {table} - EXISTS")
        except Exception as e:
            error_str = str(e)
            if 'PGRST205' in error_str or 'not find' in error_str:
                print(f"  ❌ {table} - DOES NOT EXIST")
            else:
                print(f"  ⚠️ {table} - ERROR: {error_str[:100]}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Go to Supabase Dashboard → SQL Editor")
print("2. Run the migration SQL from: migrations/001_create_matches_table.sql")
print("3. This will create:")
print("   - matches table")
print("   - product_listings table")
print("   - service_listings table")
print("   - mutual_listings table")
print("   - All necessary indexes")
print("\n4. Then rerun: python3 test_complete_flow.py")
print("\n" + "="*80)
