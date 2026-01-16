

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
        from embedding_builder import build_embedding_text
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
