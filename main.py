from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import uuid

# Import project modules
from schema_normalizer_v2 import normalize_and_validate_v2
from ingestion_pipeline import IngestionClients, ingest_listing
from retrieval_service import RetrievalClients, retrieve_candidates
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import uuid

# Import project modules
from schema_normalizer_v2 import normalize_and_validate_v2
from ingestion_pipeline import IngestionClients, ingest_listing
from retrieval_service import RetrievalClients, retrieve_candidates
from listing_matcher_v2 import listing_matches_v2
from embedding_builder import build_embedding_text
import numpy as np

app = FastAPI(title="Vriddhi Matching Engine API", version="2.0")

# Global clients
ingestion_clients = IngestionClients()
retrieval_clients = RetrievalClients()
is_initialized = False
init_error = None

async def initialize_services():
    """Run initialization in a background thread to allow instant server startup."""
    global is_initialized, init_error
    print("⏳ Starting background initialization...")
    try:
        if os.environ.get("SUPABASE_URL"):
            # Run blocking init calls in a separate thread
            await asyncio.to_thread(ingestion_clients.initialize)
            await asyncio.to_thread(retrieval_clients.initialize)
            is_initialized = True
            print("✅ Clients initialized successfully in background")
        else:
            print("⚠️ Warning: SUPABASE_URL not set. Clients not initialized.")
    except Exception as e:
        init_error = str(e)
        print(f"❌ Error initializing clients: {e}")

@app.on_event("startup")
async def startup_event():
    # Start initialization as a background task
    asyncio.create_task(initialize_services())

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

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "initialized": is_initialized, 
        "service": "Vriddhi Matching Engine V2"
    }

@app.get("/health")
def health_check():
    """Simple health check for Render"""
    return {"status": "ok"}

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
    try:
        listing_old = normalize_and_validate_v2(request.listing)
        return {"status": "success", "normalized_listing": listing_old}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
