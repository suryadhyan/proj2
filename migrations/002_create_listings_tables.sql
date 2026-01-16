-- ============================================================================
-- Migration: Create listings tables (product, service, mutual)
-- Date: 2026-01-16
-- Purpose: Create the three listings tables needed for /store-listing endpoint
-- ============================================================================

-- ============================================================================
-- PART 1: Create product_listings table
-- ============================================================================

CREATE TABLE IF NOT EXISTS product_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for product_listings
CREATE INDEX IF NOT EXISTS idx_product_user ON product_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_product_created ON product_listings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_product_match ON product_listings(match_id);

-- ============================================================================
-- PART 2: Create service_listings table
-- ============================================================================

CREATE TABLE IF NOT EXISTS service_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for service_listings
CREATE INDEX IF NOT EXISTS idx_service_user ON service_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_service_created ON service_listings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_match ON service_listings(match_id);

-- ============================================================================
-- PART 3: Create mutual_listings table
-- ============================================================================

CREATE TABLE IF NOT EXISTS mutual_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for mutual_listings
CREATE INDEX IF NOT EXISTS idx_mutual_user ON mutual_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_mutual_created ON mutual_listings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mutual_match ON mutual_listings(match_id);

-- ============================================================================
-- VERIFICATION QUERIES (Run after migration)
-- ============================================================================

-- Check if all tables exist
-- SELECT table_name FROM information_schema.tables
-- WHERE table_name IN ('product_listings', 'service_listings', 'mutual_listings')
-- ORDER BY table_name;

-- Check table structures
-- SELECT table_name, column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name IN ('product_listings', 'service_listings', 'mutual_listings')
-- ORDER BY table_name, ordinal_position;

-- Check indexes
-- SELECT tablename, indexname FROM pg_indexes
-- WHERE tablename IN ('product_listings', 'service_listings', 'mutual_listings')
-- ORDER BY tablename, indexname;
