import os, re, json, math, hashlib, uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import fitz

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UPLOADS = ROOT / "uploads"
DATA.mkdir(exist_ok=True); UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="AEC Proposal Intelligence Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

class GenerateIn(BaseModel):
    requirement_id: str
    tone: str = "confident"

class SaveDraftIn(BaseModel):
    requirement_id: str
    text: str

class AnalyzeTextIn(BaseModel):
    text: str
    name: str = "Pasted RFP"


def read_json(name, default):
    p = DATA / name
    if not p.exists(): return default
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(name, value):
    (DATA / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

def tokens(text):
    return re.findall(r"[a-z0-9]{2,}", text.lower())

def embed(text, dims=256):
    v = [0.0] * dims
    for tok in tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dims] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def score(a, b):
    return sum(x*y for x, y in zip(a, b))

def retrieve(query, k=3):
    q = embed(query)
    items = read_json("experience.json", [])
    ranked = sorted(items, key=lambda x: score(q, embed(" ".join([x.get("title",""), x.get("summary",""), " ".join(x.get("services",[])), " ".join(x.get("outcomes",[]))]))), reverse=True)
    return ranked[:k]

def extract_requirements(pages):
    reqs, seen = [], set()
    pattern = re.compile(r"\b(shall|must|required|will provide|should|submit|demonstrate|include|minimum|no later than)\b", re.I)
    for page_no, text in pages:
        lines = [re.sub(r"^[\s•●▪◦\-–—\d.)]+", "", x).strip() for x in text.splitlines()]
        candidates = []
        for line in lines:
            if 18 <= len(line) <= 500 and pattern.search(line): candidates.append(line)
        for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            sent = re.sub(r"\s+", " ", sent).strip()
            if 25 <= len(sent) <= 500 and pattern.search(sent): candidates.append(sent)
        for c in candidates:
            key = re.sub(r"\W", "", c.lower())[:160]
            if key in seen: continue
            seen.add(key)
            lower = c.lower()
            if any(w in lower for w in ["deadline", "no later", "due date", "submit by"]): cat = "Submission"
            elif any(w in lower for w in ["experience", "project", "reference", "qualification"]): cat = "Experience"
            elif any(w in lower for w in ["fee", "price", "cost", "budget"]): cat = "Commercial"
            elif any(w in lower for w in ["safety", "insurance", "license", "compliance"]): cat = "Compliance"
            else: cat = "Technical"
            reqs.append({"id": f"REQ-{len(reqs)+1:03d}", "text": c, "page": page_no, "category": cat, "priority": "High" if re.search(r"\bshall\b|\bmust\b|\brequired\b", lower) else "Medium", "status": "Open"})
    if not reqs:
        for page_no, text in pages:
            for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
                sent = sent.strip()
                if 35 <= len(sent) <= 360:
                    reqs.append({"id": f"REQ-{len(reqs)+1:03d}", "text": sent, "page": page_no, "category": "Technical", "priority": "Review", "status": "Open"})
                    if len(reqs) >= 12: break
            if len(reqs) >= 12: break
    return reqs[:80]

async def openrouter(messages, temperature=0.25):
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key: return None, "Demo mode: add OPENROUTER_API_KEY to .env for AI generation."
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost:8000", "X-Title": "AEC Proposal Intelligence Agent"}
    payload = {"model": os.getenv("OPENROUTER_MODEL", "openrouter/free"), "messages": messages, "temperature": temperature, "max_tokens": 900}
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"OpenRouter unavailable: {str(e)[:160]}"

def current():
    return read_json("workspace.json", {"document": None, "requirements": [], "drafts": {}})

@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/workspace")
def workspace(): return current()

@app.get("/api/experience")
def experience(): return read_json("experience.json", [])

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Only PDF files are supported in this MVP.")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024: raise HTTPException(413, "PDF exceeds the 25 MB limit.")
    path = UPLOADS / f"{uuid.uuid4().hex}.pdf"; path.write_bytes(raw)
    try:
        doc = fitz.open(path)
        pages = [(i+1, p.get_text("text")) for i, p in enumerate(doc)]
    except Exception as e: raise HTTPException(400, f"Could not read PDF: {e}")
    reqs = extract_requirements(pages)
    ws = {"document": {"name": file.filename, "pages": len(pages), "characters": sum(len(x[1]) for x in pages)}, "requirements": reqs, "drafts": {}}
    write_json("workspace.json", ws)
    return ws

@app.post("/api/analyze-text")
def analyze_text(body: AnalyzeTextIn):
    pages = [(1, body.text)]
    ws = {"document": {"name": body.name, "pages": 1, "characters": len(body.text)}, "requirements": extract_requirements(pages), "drafts": {}}
    write_json("workspace.json", ws)
    return ws

@app.post("/api/demo")
def demo():
    text = (DATA / "demo_rfp.txt").read_text(encoding="utf-8")
    ws = {"document": {"name": "Riverfront Civic Center RFP.pdf", "pages": 18, "characters": len(text)}, "requirements": extract_requirements([(2, text[:1600]), (5, text[1600:3200]), (9, text[3200:])]), "drafts": {}}
    write_json("workspace.json", ws)
    return ws

@app.post("/api/generate")
async def generate(body: GenerateIn):
    ws = current(); req = next((r for r in ws["requirements"] if r["id"] == body.requirement_id), None)
    if not req: raise HTTPException(404, "Requirement not found")
    evidence = retrieve(req["text"], 3)
    evidence_text = "\n\n".join(f"[{x['id']}] {x['title']}\n{x['summary']}\nServices: {', '.join(x['services'])}\nOutcomes: {'; '.join(x['outcomes'])}" for x in evidence)
    system = "You are an expert AEC proposal writer. Draft only supportable claims. Cite every project-specific claim with the supplied evidence ID in square brackets. Never invent metrics, clients, certifications, staff, or outcomes. If evidence is insufficient, write [EVIDENCE NEEDED]."
    user = f"RFP requirement:\n{req['text']}\n\nApproved experience records:\n{evidence_text}\n\nWrite a concise, {body.tone} response of 140-220 words. Start with a direct compliance statement, then approach and proof."
    answer, warning = await openrouter([{"role":"system","content":system},{"role":"user","content":user}])
    if not answer:
        top = evidence[0]
        answer = f"We will meet this requirement through a disciplined delivery plan that aligns scope, schedule, quality reviews, and accountable ownership. Our team will confirm acceptance criteria at kickoff, maintain a requirement-level action register, and document each review before submission.\n\nRelevant experience includes {top['title']}, where our team provided {', '.join(top['services'][:3])}. {top['outcomes'][0]} [{top['id']}]. This record demonstrates an applicable delivery approach; project-specific staffing, dates, and commitments will be confirmed before final submission.\n\n[EVIDENCE NEEDED: Confirm proposed personnel and client-approved performance metrics.]"
    ws["drafts"][req["id"]] = {"text": answer, "evidence": evidence, "warning": warning}
    req["status"] = "Drafted"; write_json("workspace.json", ws)
    return ws["drafts"][req["id"]]

@app.post("/api/save-draft")
def save_draft(body: SaveDraftIn):
    ws = current(); existing = ws["drafts"].get(body.requirement_id, {})
    existing["text"] = body.text; ws["drafts"][body.requirement_id] = existing
    for r in ws["requirements"]:
        if r["id"] == body.requirement_id: r["status"] = "Drafted"
    write_json("workspace.json", ws); return {"ok": True}

@app.get("/api/compliance")
def compliance():
    ws = current(); rows=[]
    for r in ws["requirements"]:
        draft = ws["drafts"].get(r["id"], {}).get("text", "")
        cited = bool(re.search(r"\[EXP-\d+\]", draft))
        unsupported = bool(draft) and not cited
        missing = "EVIDENCE NEEDED" in draft or not draft
        status = "Ready" if draft and cited and not missing else ("Review" if draft else "Missing")
        rows.append({**r, "status": status, "has_draft": bool(draft), "has_citation": cited, "unsupported_claim_risk": unsupported, "missing_evidence": missing})
    return {"rows": rows, "ready": sum(x["status"]=="Ready" for x in rows), "review": sum(x["status"]=="Review" for x in rows), "missing": sum(x["status"]=="Missing" for x in rows)}

@app.get("/api/export")
def export():
    ws = current(); lines=[f"# Proposal Response — {ws.get('document',{}).get('name','Untitled')}\n"]
    for r in ws["requirements"]:
        lines += [f"## {r['id']} · {r['category']}", f"**Requirement (p. {r['page']}):** {r['text']}", "", ws["drafts"].get(r["id"],{}).get("text","_No response drafted._"), ""]
    return PlainTextResponse("\n".join(lines), media_type="text/markdown", headers={"Content-Disposition":"attachment; filename=proposal-response.md"})
