from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import uuid
import json
from openai import OpenAI
import asyncio
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import project modules
from schema_normalizer_v2 import normalize_and_validate_v2
from ingestion_pipeline import IngestionClients, ingest_listing
from retrieval_service import RetrievalClients, retrieve_candidates
from listing_matcher_v2 import listing_matches_v2
from embedding_builder import build_embedding_text

app = FastAPI(title="Vriddhi Matching Engine API", version="2.0")

# Global clients
ingestion_clients = IngestionClients()
retrieval_clients = RetrievalClients()
openai_client = None
extraction_prompt = None
is_initialized = False
init_error = None

# Load extraction prompt
def load_extraction_prompt():
    """Load the extraction prompt from file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt", "GLOBAL_REFERENCE_CONTEXT.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Warning: Could not load extraction prompt: {e}")
        return None

# Initialize OpenAI client
def initialize_openai():
    """Initialize OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    else:
        print("⚠️ Warning: OPENAI_API_KEY not set. Extraction endpoint will not work.")
        return None

async def initialize_services():
    """Run initialization in a background thread to allow instant server startup."""
    global is_initialized, init_error, openai_client, extraction_prompt
    print("⏳ Starting background initialization...")
    print(f"🌍 SUPABASE_URL: {'SET' if os.environ.get('SUPABASE_URL') else 'NOT SET'}")
    print(f"🔑 OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")

    try:
        # Initialize OpenAI client (fast, non-blocking)
        print("📝 Initializing OpenAI client...")
        openai_client = initialize_openai()
        print("✅ OpenAI client ready")

        # Load extraction prompt (fast, non-blocking)
        print("📄 Loading extraction prompt...")
        extraction_prompt = load_extraction_prompt()
        print(f"✅ Extraction prompt loaded ({len(extraction_prompt) if extraction_prompt else 0} chars)")

        if os.environ.get("SUPABASE_URL"):
            # Run heavy init calls in a separate thread
            print("🔄 Initializing ingestion clients (in background)...")
            await asyncio.to_thread(ingestion_clients.initialize)
            print("✅ Ingestion clients initialized")

            print("🔄 Initializing retrieval clients (in background)...")
            await asyncio.to_thread(retrieval_clients.initialize)
            print("✅ Retrieval clients initialized")

            is_initialized = True
            print("✅ ALL clients initialized successfully")
        else:
            print("⚠️ SUPABASE_URL not set. Skipping database/vector clients.")
            is_initialized = True  # Mark as initialized for extraction endpoints
            print("✅ Server ready (extraction-only mode)")
    except Exception as e:
        init_error = str(e)
        print(f"❌ Error initializing clients: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    """Start server immediately, run initialization in background."""
    print("🚀 FastAPI server starting...")
    print(f"📍 Server should be available on port {os.environ.get('PORT', '8000')}")
    # Start initialization as a fire-and-forget background task
    asyncio.create_task(initialize_services())
    print("✅ Server startup complete (initialization running in background)")

def check_service_health():
    """Helper to check if services are ready."""
    if init_error:
        raise HTTPException(status_code=500, detail=f"Service initialization failed: {init_error}")
    if not is_initialized:
        raise HTTPException(status_code=503, detail="Service is still starting up (loading models). Please try again in 30 seconds.")

def semantic_implies(candidate_val: str, required_val: str) -> bool:
    """
    Check if candidate_val semantically implies required_val using embeddings.
    """
    if not ingestion_clients.embedding_model:
        return candidate_val.lower() == required_val.lower()
        
    v1 = ingestion_clients.embedding_model.encode(candidate_val)
    v2 = ingestion_clients.embedding_model.encode(required_val)
    
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(sim) > 0.82

class ListingRequest(BaseModel):
    listing: Dict[str, Any]

class MatchRequest(BaseModel):
    listing_a: Dict[str, Any]
    listing_b: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str

class DualQueryRequest(BaseModel):
    query_a: str
    query_b: str

class SearchAndMatchRequest(BaseModel):
    query: str
    user_id: str

class StoreListingRequest(BaseModel):
    listing_json: Dict[str, Any]
    user_id: str
    match_id: Optional[str] = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "initialized": is_initialized,
        "service": "Vriddhi Matching Engine V2"
    }

@app.get("/health")
def health_check():
    """Simple health check for Render - responds immediately"""
    return {"status": "ok"}

@app.get("/ping")
def ping():
    """Ultra-simple ping endpoint"""
    return "pong"

@app.post("/ingest")
async def ingest_endpoint(request: ListingRequest):
    check_service_health()
    try:
        # 1. Normalize
        listing_old = normalize_and_validate_v2(request.listing)
        
        # 2. Ingest
        listing_id, _ = ingest_listing(ingestion_clients, listing_old, verbose=True)
        
        return {
            "status": "success",
            "listing_id": listing_id,
            "message": "Listing normalized and ingested successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_endpoint(request: ListingRequest, limit: int = 10):
    check_service_health()
    try:
        # 1. Normalize
        listing_old = normalize_and_validate_v2(request.listing)

        # 2. Retrieve
        candidate_ids = retrieve_candidates(retrieval_clients, listing_old, limit=limit, verbose=True)

        return {
            "status": "success",
            "count": len(candidate_ids),
            "candidates": candidate_ids
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error in /search endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match")
async def match_endpoint(request: MatchRequest):
    check_service_health()
    try:
        # 1. Normalize
        listing_a_old = normalize_and_validate_v2(request.listing_a)
        listing_b_old = normalize_and_validate_v2(request.listing_b)
        
        # 2. Match with semantic implication
        is_match = listing_matches_v2(listing_a_old, listing_b_old, implies_fn=semantic_implies)
        
        return {
            "status": "success",
            "match": is_match,
            "details": "Semantic match successful" if is_match else "No match found"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/normalize")
async def normalize_endpoint(request: ListingRequest):
    """
    Helper endpoint to just normalize a listing (NEW -> OLD) without ingesting.
    """
    try:
        listing_old = normalize_and_validate_v2(request.listing)
        return {"status": "success", "normalized_listing": listing_old}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# NEW: GPT EXTRACTION ENDPOINTS
# ============================================================================

def extract_from_query(query: str) -> Dict[str, Any]:
    """
    Extract structured NEW schema from natural language query using GPT API.

    Args:
        query: Natural language query (e.g., "need a plumber who speaks kannada")

    Returns:
        Structured NEW schema dictionary

    Raises:
        HTTPException: If OpenAI client not initialized or API call fails
    """
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured. Set OPENAI_API_KEY environment variable."
        )

    if not extraction_prompt:
        raise HTTPException(
            status_code=500,
            detail="Extraction prompt not loaded. Check prompt/GLOBAL_REFERENCE_CONTEXT.md exists."
        )

    try:
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # Parse response
        output_text = response.choices[0].message.content
        extracted_data = json.loads(output_text)

        return extracted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract")
async def extract_endpoint(request: QueryRequest):
    """
    NEW ENDPOINT: Extract structured schema from natural language query.

    Input: Natural language query
    Output: Structured NEW schema (14 fields, axis-based)

    Example:
        POST /extract
        {
            "query": "need a plumber who speaks kannada"
        }

        Returns:
        {
            "status": "success",
            "query": "need a plumber who speaks kannada",
            "extracted_listing": {
                "intent": "service",
                "subintent": "seek",
                "domain": ["construction & trades"],
                "items": [{"type": "plumbing", ...}],
                "other_party_preferences": {
                    "categorical": {"language": "kannada"},
                    ...
                },
                ...
            }
        }
    """
    try:
        extracted_listing = extract_from_query(request.query)

        return {
            "status": "success",
            "query": request.query,
            "extracted_listing": extracted_listing
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-and-normalize")
async def extract_and_normalize_endpoint(request: QueryRequest):
    """
    NEW ENDPOINT: Extract from natural language, then normalize to OLD schema.

    This combines:
    1. GPT extraction (natural language -> NEW schema)
    2. Schema normalization (NEW schema -> OLD schema)

    Input: Natural language query
    Output: OLD schema format ready for matching
    """
    try:
        # Step 1: Extract NEW schema
        extracted_listing = extract_from_query(request.query)

        # Step 2: Normalize to OLD schema
        normalized_listing = normalize_and_validate_v2(extracted_listing)

        return {
            "status": "success",
            "query": request.query,
            "extracted_listing": extracted_listing,
            "normalized_listing": normalized_listing
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-and-match")
async def extract_and_match_endpoint(request: DualQueryRequest):
    """
    NEW ENDPOINT: Extract from TWO natural language queries and match them.

    Complete pipeline:
    1. Extract listing A from query_a (GPT)
    2. Extract listing B from query_b (GPT)
    3. Normalize both (NEW -> OLD)
    4. Match them (semantic matching)

    Input: Two natural language queries
    Output: Match result (true/false) with details

    Example:
        POST /extract-and-match
        {
            "query_a": "need a plumber who speaks kannada",
            "query_b": "I am a plumber, I speak kannada and english"
        }

        Returns:
        {
            "status": "success",
            "query_a": "...",
            "query_b": "...",
            "match": true,
            "details": "Semantic match successful"
        }
    """
    check_service_health()
    try:
        # Step 1: Extract both queries
        extracted_a = extract_from_query(request.query_a)
        extracted_b = extract_from_query(request.query_b)

        # Step 2: Normalize both
        listing_a_old = normalize_and_validate_v2(extracted_a)
        listing_b_old = normalize_and_validate_v2(extracted_b)

        # Step 3: Match with semantic implication
        is_match = listing_matches_v2(listing_a_old, listing_b_old, implies_fn=semantic_implies)

        return {
            "status": "success",
            "query_a": request.query_a,
            "query_b": request.query_b,
            "extracted_a": extracted_a,
            "extracted_b": extracted_b,
            "normalized_a": listing_a_old,
            "normalized_b": listing_b_old,
            "match": is_match,
            "details": "Semantic match successful" if is_match else "No match found"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEW: SEARCH AND MATCH + STORE LISTING ENDPOINTS
# ============================================================================

@app.post("/search-and-match")
async def search_and_match_endpoint(request: SearchAndMatchRequest):
    """
    NEW ENDPOINT: Complete search and match flow with history storage.

    Flow:
    1. Extract structured JSON from natural language query (GPT)
    2. Search database for matching listings (Qdrant + SQL)
    3. Boolean match each candidate (listing_matches_v2)
    4. Store EVERYTHING in matches table (query + results)
    5. Return matches and query_json

    This endpoint ALWAYS stores search history (even if 0 matches found).

    Input:
        - query: Natural language query
        - user_id: User performing the search

    Output:
        - match_id: UUID of matches table entry
        - query_text: Original query
        - query_json: GPT extracted JSON
        - has_matches: True/False
        - match_count: Number of matches
        - matched_listings: Full details of matched listings
    """
    check_service_health()

    try:
        # Step 1: GPT Extraction
        print(f"\n🔍 Search and Match for user: {request.user_id}")
        print(f"📝 Query: {request.query}")

        extracted_json = extract_from_query(request.query)

        # Step 2: Normalize
        normalized_query = normalize_and_validate_v2(extracted_json)

        # Step 3: Search database for candidates
        print(f"🔎 Searching database...")
        candidate_ids = retrieve_candidates(
            retrieval_clients,
            normalized_query,
            limit=100,
            verbose=True
        )

        print(f"📊 Found {len(candidate_ids)} candidates")

        # Step 4: Boolean match each candidate
        matched_listings = []
        matched_user_ids = []
        matched_listing_ids = []

        if candidate_ids:
            # Fetch candidates from Supabase
            intent = normalized_query.get("intent")
            table_name = f"{intent}_listings"

            print(f"🔍 Fetching candidates from {table_name}...")

            for listing_id in candidate_ids:
                try:
                    # Fetch from Supabase
                    response = ingestion_clients.supabase.table(table_name).select("*").eq("id", listing_id).execute()

                    if response.data and len(response.data) > 0:
                        candidate_row = response.data[0]
                        candidate_data = candidate_row["data"]
                        candidate_user_id = candidate_row.get("user_id")

                        # Boolean match
                        is_match = listing_matches_v2(
                            normalized_query,
                            candidate_data,
                            implies_fn=semantic_implies
                        )

                        if is_match:
                            matched_listings.append({
                                "listing_id": listing_id,
                                "user_id": candidate_user_id,
                                "data": candidate_data
                            })
                            if candidate_user_id:
                                matched_user_ids.append(candidate_user_id)
                            matched_listing_ids.append(listing_id)

                except Exception as e:
                    print(f"⚠️ Error fetching/matching listing {listing_id}: {e}")
                    continue

        # Step 5: Store in matches table
        match_id = str(uuid.uuid4())
        has_matches = len(matched_listings) > 0
        match_count = len(matched_listings)

        matches_data = {
            "match_id": match_id,
            "query_user_id": request.user_id,
            "query_text": request.query,
            "query_json": extracted_json,
            "has_matches": has_matches,
            "match_count": match_count,
            "matched_user_ids": matched_user_ids,
            "matched_listing_ids": matched_listing_ids
        }

        print(f"💾 Storing search history in matches table...")
        ingestion_clients.supabase.table("matches").insert(matches_data).execute()
        print(f"✅ Stored with match_id: {match_id}")

        return {
            "status": "success",
            "match_id": match_id,
            "query_text": request.query,
            "query_json": extracted_json,
            "has_matches": has_matches,
            "match_count": match_count,
            "matched_listings": matched_listings,
            "message": f"Found {match_count} matches" if has_matches else "No matches found. You can store your query for future matching."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in search-and-match: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-and-match-direct")
async def search_and_match_direct_endpoint(request: StoreListingRequest):
    """
    TESTING ENDPOINT: Complete search and match flow with pre-formatted JSON (bypasses GPT).

    Flow:
    1. Accept pre-formatted JSON directly (skip GPT extraction)
    2. Search database for matching listings (Qdrant + SQL)
    3. Boolean match each candidate (listing_matches_v2)
    4. Return matches (does NOT store search history)

    This endpoint is for testing only - bypasses GPT extraction.

    Input:
        - listing_json: Complete listing JSON (NEW schema format)
        - user_id: User performing the search

    Output:
        - has_matches: True/False
        - match_count: Number of matches
        - matches: List of matched user_ids or listing objects
    """
    check_service_health()

    try:
        # Step 1: Normalize (skip GPT extraction)
        normalized_query = normalize_and_validate_v2(request.listing_json)

        # Step 2: Search database for candidates
        candidate_ids = retrieve_candidates(
            retrieval_clients,
            normalized_query,
            limit=100,
            verbose=False
        )

        # Step 3: Boolean match each candidate
        matched_listings = []
        matched_user_ids = []
        seen_user_ids = set()  # Track unique user_ids to avoid duplicates

        intent = normalized_query.get("intent")
        table_name = f"{intent}_listings"

        for listing_id in candidate_ids:
            try:
                # Fetch candidate listing from database
                response = ingestion_clients.supabase.table(table_name).select("*").eq("id", listing_id).execute()

                if response.data and len(response.data) > 0:
                    row = response.data[0]
                    candidate_data = row.get("data")
                    candidate_user_id = row.get("user_id")

                    # Skip if we've already matched this user
                    if candidate_user_id in seen_user_ids:
                        continue

                    # Run boolean match
                    is_match = listing_matches_v2(normalized_query, candidate_data, implies_fn=semantic_implies)

                    if is_match:
                        matched_listings.append({
                            "listing_id": listing_id,
                            "user_id": candidate_user_id,
                            "data": candidate_data
                        })
                        if candidate_user_id:
                            matched_user_ids.append(candidate_user_id)
                            seen_user_ids.add(candidate_user_id)  # Mark as seen

            except Exception as e:
                print(f"⚠️ Error fetching/matching listing {listing_id}: {e}")
                continue

        has_matches = len(matched_listings) > 0
        match_count = len(matched_listings)

        return {
            "status": "success",
            "has_matches": has_matches,
            "match_count": match_count,
            "matches": matched_user_ids,  # Return user_ids for compatibility with test
            "matched_listings": matched_listings
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in search-and-match-direct: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store-listing")
async def store_listing_endpoint(request: StoreListingRequest):
    """
    NEW ENDPOINT: Store listing in database for future matching.

    Flow:
    1. Validate listing JSON
    2. Normalize to OLD format
    3. Store in appropriate listings table (with user_id and optional match_id)
    4. Generate embedding
    5. Store embedding in Qdrant
    6. Return listing_id

    This endpoint ONLY stores. It does NOT search or match.

    Input:
        - listing_json: Complete listing JSON (NEW schema format)
        - user_id: User who owns this listing
        - match_id: Optional reference to matches table (if from search)

    Output:
        - listing_id: UUID of stored listing
        - intent: Product/Service/Mutual
        - message: Confirmation
    """
    check_service_health()

    try:
        print(f"\n💾 Store Listing for user: {request.user_id}")

        # Step 1: Validate and normalize
        normalized_listing = normalize_and_validate_v2(request.listing_json)

        # Step 2: Ingest (stores in Supabase + Qdrant)
        listing_id = str(uuid.uuid4())

        # Get intent for table selection
        intent = normalized_listing.get("intent")
        if not intent:
            raise ValueError("Listing missing 'intent' field")

        table_name = f"{intent}_listings"

        # Prepare data with user_id and match_id
        data = {
            "id": listing_id,
            "user_id": request.user_id,
            "match_id": request.match_id,
            "data": normalized_listing
        }

        print(f"📝 Storing in {table_name}...")
        ingestion_clients.supabase.table(table_name).insert(data).execute()
        print(f"✅ Stored in Supabase with listing_id: {listing_id}")

        # Step 3: Generate and store embedding in Qdrant
        embedding_text = build_embedding_text(normalized_listing)
        embedding = ingestion_clients.embedding_model.encode(embedding_text).tolist()

        # Select Qdrant collection
        collection_name = f"{intent}_vectors"

        # Build payload
        payload = {
            "listing_id": listing_id,
            "intent": intent
        }

        if intent in ["product", "service"]:
            payload["domain"] = normalized_listing.get("domain", [])
        elif intent == "mutual":
            payload["category"] = normalized_listing.get("category", [])

        # Store in Qdrant
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=listing_id,
            vector=embedding,
            payload=payload
        )

        print(f"🔢 Storing embedding in {collection_name}...")
        ingestion_clients.qdrant.upsert(
            collection_name=collection_name,
            points=[point]
        )
        print(f"✅ Stored in Qdrant")

        return {
            "status": "success",
            "listing_id": listing_id,
            "user_id": request.user_id,
            "intent": intent,
            "match_id": request.match_id,
            "message": f"Listing stored successfully. It will be visible to future searches."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in store-listing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
