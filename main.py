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

def semantic_implies(candidate_val: str, required_val: str) -> bool:
    """
    Check if candidate_val semantically implies required_val using embeddings.
    Threshold of 0.82 indicates high semantic similarity.
    """
    if not ingestion_clients.embedding_model:
        return candidate_val.lower() == required_val.lower()
        
    # Generate embeddings for both terms
    v1 = ingestion_clients.embedding_model.encode(candidate_val)
    v2 = ingestion_clients.embedding_model.encode(required_val)
    
    # Cosine similarity
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(sim) > 0.82

@app.on_event("startup")
async def startup_event():
    # Initialize clients on startup if env vars are present
    try:
        if os.environ.get("SUPABASE_URL"):
            ingestion_clients.initialize()
            retrieval_clients.initialize()
            print("Clients initialized successfully")
        else:
            print("Warning: SUPABASE_URL not set. Clients not initialized.")
    except Exception as e:
        print(f"Error initializing clients: {e}")

class ListingRequest(BaseModel):
    listing: Dict[str, Any]

class MatchRequest(BaseModel):
    listing_a: Dict[str, Any]
    listing_b: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Vriddhi Matching Engine V2"}

@app.post("/ingest")
async def ingest_endpoint(request: ListingRequest):
    """
    Ingest a NEW schema listing.
    1. Normalize to OLD format
    2. Ingest to Supabase + Qdrant
    """
    try:
        # 1. Normalize
        listing_old = normalize_and_validate_v2(request.listing)
        
        # 2. Ingest
        if not ingestion_clients.supabase:
             raise HTTPException(status_code=503, detail="Ingestion service not initialized (check env vars)")
             
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
    """
    Search for candidates using a NEW schema listing as query.
    1. Normalize to OLD format
    2. Retrieve candidates from Qdrant/Supabase
    """
    try:
        # 1. Normalize
        listing_old = normalize_and_validate_v2(request.listing)
        
        # 2. Retrieve
        if not retrieval_clients.qdrant:
             raise HTTPException(status_code=503, detail="Retrieval service not initialized (check env vars)")

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
    """
    Check if two NEW schema listings match.
    1. Normalize both
    2. Run boolean matching logic with semantic implication
    """
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
