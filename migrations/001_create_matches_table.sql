-- ============================================================================
-- Migration: Create matches table and update listings tables
-- Date: 2026-01-16
-- Purpose: Add search history tracking and user association
-- ============================================================================

-- ============================================================================
-- PART 1: Create matches table
-- ============================================================================

CREATE TABLE IF NOT EXISTS matches (
    match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_user_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    query_json JSONB NOT NULL,
    has_matches BOOLEAN NOT NULL,
    match_count INTEGER NOT NULL DEFAULT 0,
    matched_user_ids UUID[] DEFAULT '{}',
    matched_listing_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_matches_user ON matches(query_user_id);
CREATE INDEX IF NOT EXISTS idx_matches_created ON matches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_matches_has_matches ON matches(has_matches);

-- ============================================================================
-- PART 2: Update product_listings table
-- ============================================================================

-- Add user_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'product_listings' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE product_listings ADD COLUMN user_id UUID;
    END IF;
END $$;

-- Add match_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'product_listings' AND column_name = 'match_id'
    ) THEN
        ALTER TABLE product_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_product_user ON product_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_product_match ON product_listings(match_id);

-- ============================================================================
-- PART 3: Update service_listings table
-- ============================================================================

-- Add user_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'service_listings' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE service_listings ADD COLUMN user_id UUID;
    END IF;
END $$;

-- Add match_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'service_listings' AND column_name = 'match_id'
    ) THEN
        ALTER TABLE service_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_service_user ON service_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_service_match ON service_listings(match_id);

-- ============================================================================
-- PART 4: Update mutual_listings table
-- ============================================================================

-- Add user_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'mutual_listings' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE mutual_listings ADD COLUMN user_id UUID;
    END IF;
END $$;

-- Add match_id column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'mutual_listings' AND column_name = 'match_id'
    ) THEN
        ALTER TABLE mutual_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_mutual_user ON mutual_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_mutual_match ON mutual_listings(match_id);

-- ============================================================================
-- VERIFICATION QUERIES (Run these after migration)
-- ============================================================================

-- Check if matches table exists
-- SELECT table_name FROM information_schema.tables WHERE table_name = 'matches';

-- Check matches table structure
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'matches'
-- ORDER BY ordinal_position;

-- Check if user_id and match_id columns exist in listings tables
-- SELECT table_name, column_name
-- FROM information_schema.columns
-- WHERE table_name IN ('product_listings', 'service_listings', 'mutual_listings')
-- AND column_name IN ('user_id', 'match_id')
-- ORDER BY table_name, column_name;

-- Check indexes
-- SELECT tablename, indexname FROM pg_indexes
-- WHERE tablename IN ('matches', 'product_listings', 'service_listings', 'mutual_listings')
-- ORDER BY tablename, indexname;
