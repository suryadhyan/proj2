# 🚀 Deployment Guide - GPT API Integration

**Date:** 2026-01-15
**Status:** Ready for deployment

---

## ✅ What Was Integrated

### GPT API extraction has been added to `main.py`:

1. **OpenAI client initialization** - Loads during startup
2. **Extraction prompt loading** - Reads `prompt/PROMPT_STAGE2.txt`
3. **3 New endpoints** - For natural language extraction

### New Endpoints:

#### 1. `/extract` - Extract structured data
```bash
POST /extract
{
  "query": "need a plumber who speaks kannada"
}

Returns:
{
  "status": "success",
  "query": "...",
  "extracted_listing": {
    "intent": "service",
    "subintent": "seek",
    "other_party_preferences": {
      "categorical": {"language": "kannada"},
      ...
    },
    ...
  }
}
```

#### 2. `/extract-and-normalize` - Extract + Normalize
```bash
POST /extract-and-normalize
{
  "query": "need a plumber who speaks kannada"
}

Returns both NEW schema (from GPT) and OLD schema (for matching)
```

#### 3. `/extract-and-match` - Complete pipeline
```bash
POST /extract-and-match
{
  "query_a": "need a plumber who speaks kannada",
  "query_b": "I am a plumber, I speak kannada"
}

Returns:
{
  "status": "success",
  "match": true,
  "extracted_a": {...},
  "extracted_b": {...},
  "normalized_a": {...},
  "normalized_b": {...},
  "details": "Semantic match successful"
}
```

### Existing Endpoints (UNCHANGED):
- ✅ `/ingest` - Still works
- ✅ `/search` - Still works
- ✅ `/match` - Still works
- ✅ `/normalize` - Still works

---

## 🔧 Changes Made to Fix Render Timeout

### Problem:
```
Port scan timeout reached, no open ports detected
```

### Solution Applied:

1. **Added verbose logging** to track initialization progress
2. **Made server start IMMEDIATELY** - initialization runs in background
3. **Added `/ping` endpoint** - Ultra-simple endpoint that responds instantly
4. **Added error handling** - Prints stack traces if initialization fails

### Key Changes in `main.py`:

```python
# New imports
from openai import OpenAI
import json

# New global variables
openai_client = None
extraction_prompt = None

# New functions
def load_extraction_prompt()  # Loads prompt file
def initialize_openai()       # Initializes OpenAI client
def extract_from_query(query) # Core extraction logic

# Updated startup with logging
@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI server starting...")
    asyncio.create_task(initialize_services())  # Non-blocking
    print("✅ Server startup complete")
```

---

## 📋 Environment Variables Needed

### For Extraction Endpoints (NEW):
```bash
OPENAI_API_KEY=sk-...  # Required for /extract* endpoints
```

### For Database/Matching Endpoints (Existing):
```bash
SUPABASE_URL=https://...
SUPABASE_KEY=...
QDRANT_HOST=...
QDRANT_PORT=6333
```

**Note:** Server will start even if SUPABASE_URL is not set (extraction-only mode)

---

## 🚀 Deployment Steps

### 1. Set Environment Variable on Render:

Go to Render dashboard → Environment → Add:
```
OPENAI_API_KEY = sk-proj-...your-key...
```

### 2. Commit and Push Changes:

```bash
git add main.py DEPLOYMENT_GUIDE.md PROMPT_FIX_SUMMARY.md VERIFICATION_RESULTS.md
git commit -m "Integrate GPT API extraction into main pipeline"
git push origin main
```

### 3. Render Will Auto-Deploy:

Watch the logs for:
```
🚀 FastAPI server starting...
📍 Server should be available on port ...
✅ Server startup complete
```

### 4. Test the Deployment:

Once deployed, test with:

```bash
# Test health
curl https://proj2-qjix.onrender.com/health
# Should return: {"status":"ok"}

# Test ping
curl https://proj2-qjix.onrender.com/ping
# Should return: "pong"

# Test extraction (replace with your query)
curl -X POST https://proj2-qjix.onrender.com/extract \
  -H "Content-Type: application/json" \
  -d '{"query":"need a plumber who speaks kannada"}'
```

---

## 🔍 Troubleshooting

### If deployment still times out:

1. **Check Build Logs:**
   - Verify `download_model.py` runs successfully
   - Check if all dependencies install

2. **Check Runtime Logs:**
   - Look for initialization messages:
     ```
     🚀 FastAPI server starting...
     📝 Initializing OpenAI client...
     ✅ OpenAI client ready
     ```

3. **Verify Environment Variables:**
   - OPENAI_API_KEY should be set
   - PORT is automatically set by Render

4. **Check for Errors:**
   - Look for `❌` in logs
   - Check stack traces

### If `/extract` endpoints don't work:

1. **Check OPENAI_API_KEY:**
   ```bash
   curl https://proj2-qjix.onrender.com/
   # Check "status": should show if initialized
   ```

2. **Check Logs for Prompt Loading:**
   ```
   📄 Loading extraction prompt...
   ✅ Extraction prompt loaded (78077 chars)
   ```

3. **Test locally first:**
   ```bash
   export OPENAI_API_KEY=sk-...
   python3 -m uvicorn main:app --reload
   # Test on localhost:8000
   ```

---

## ⚡ Performance Notes

### Initialization Time:
- **OpenAI client:** < 1 second
- **Prompt loading:** < 1 second
- **Embedding model (if SUPABASE_URL set):** 30-60 seconds (runs in background)

### API Response Times:
- `/health`, `/ping`: < 10ms
- `/extract`: ~2-5 seconds (GPT-4o API call)
- `/extract-and-match`: ~4-10 seconds (2x GPT calls + matching)

### Cost per Request:
- GPT-4o: ~$0.01-0.03 per extraction (depending on prompt size)
- For production: Use fine-tuned Mistral on Azure (much cheaper)

---

## 📊 Testing Checklist

After deployment, verify:

- [ ] Health endpoint responds: `/health` → `{"status":"ok"}`
- [ ] Ping endpoint responds: `/ping` → `"pong"`
- [ ] Root endpoint shows status: `/` → `{"status":"online", "initialized": true/false}`
- [ ] Extract endpoint works: `/extract` with sample query
- [ ] Extract-and-normalize works: `/extract-and-normalize`
- [ ] Extract-and-match works: `/extract-and-match` with 2 queries
- [ ] Existing endpoints still work: `/match`, `/normalize`

---

## 🎯 Next Steps

1. ✅ Deploy to Render with OPENAI_API_KEY
2. ✅ Test all endpoints
3. 🔄 Monitor usage and costs
4. 🔄 Collect training data for Mistral fine-tuning
5. 🔄 Deploy fine-tuned Mistral to Azure
6. 🔄 Switch from OpenAI to Azure Mistral in production

---

## 📝 Summary

**What Changed:**
- Added 3 new endpoints for GPT extraction
- Existing endpoints unchanged
- Server starts immediately (< 2 seconds)
- Background initialization for heavy operations

**What's Needed:**
- Set `OPENAI_API_KEY` environment variable
- Commit and push to trigger deployment
- Test new endpoints

**Expected Result:**
- Server deploys successfully on Render
- All endpoints (old + new) work correctly
- Natural language queries can be processed end-to-end

---

**Ready to deploy! 🚀**
