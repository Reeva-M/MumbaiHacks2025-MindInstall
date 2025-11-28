import os
import json
import time
import re
import base64
import numpy as np
import pandas as pd
import pdfplumber
import requests

from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
from datetime import datetime

# Visualizations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PDF generator
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HF_API_KEY = "Your_API_KEY"
CURRENCY_REGEX = r'(?:Rs\.?|INR|₹|USD|\$)\s?[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s?(?:crore|lakh|lac|million|billion|k|m|bn)'
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def normalize_number(token):
    if token is None:
        return None
    s = str(token).lower().replace(',', '').strip()
    try:
        if 'crore' in s:
            return float(s.replace('crore', '').strip()) * 1e7
        if 'lakh' in s or 'lac' in s:
            return float(s.replace('lakh', '').replace('lac', '').strip()) * 1e5
        if 'million' in s:
            return float(s.replace('million', '').strip()) * 1e6
        s = re.sub(r'[^0-9\.]', '', s)
        return float(s) if s else None
    except:
        return None


# ---------------------------------------------------------------------
# LLaMA CALL
# ---------------------------------------------------------------------
def call_llama(prompt, json_mode=True):
    system_msg = "You are an insurance extraction expert. Respond ONLY in valid JSON." \
        if json_mode else \
        "You are a helpful insurance policy expert. Answer based ONLY on the provided document context."

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        r = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"},
            timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except:
        return ""


# ---------------------------------------------------------------------
# JSON HELPERS
# ---------------------------------------------------------------------
def find_balanced_brace_substrings(text):
    out = []
    starts = [m.start() for m in re.finditer(r'\{', text)]
    for start in starts:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    out.append(text[start:i+1])
                    break
    return out


def repair_json(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r',\s*([\]}])', r'\1', s)
    s = re.sub(r'\bNone\b', 'null', s)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    return s


def try_parse_json(raw):
    candidates = find_balanced_brace_substrings(raw)
    for c in candidates:
        try:
            return json.loads(c)
        except:
            try:
                return json.loads(repair_json(c))
            except:
                continue
    return None


# ---------------------------------------------------------------------
# REGEX FALLBACK
# ---------------------------------------------------------------------
def regex_extract_fields(text):
    t = text.lower()
    insurer = re.search(r'(insurance company|insurer|company)[:\s-]+([a-zA-Z &]+)', t)
    insurer = insurer.group(2).title() if insurer else "Unknown"

    if 'health' in t: ptype = 'Health'
    elif 'life' in t: ptype = 'Life'
    elif 'motor' in t: ptype = 'Motor'
    elif 'home' in t: ptype = 'Home'
    else: ptype = 'Unknown'

    si = None
    m = re.search(r'(sum insured|coverage amount)[:\s-]+(' + CURRENCY_REGEX + ')', text, re.I)
    if m: si = normalize_number(m.group(2))

    prem = None
    m = re.search(r'(premium)[:\s-]+(' + CURRENCY_REGEX + ')', text, re.I)
    if m: prem = normalize_number(m.group(2))

    inc = []
    exc = []
    for ln in text.split('\n'):
        if "cover" in ln.lower() or "benefit" in ln.lower():
            inc.append(ln.strip())
        if "exclude" in ln.lower() or "not covered" in ln.lower():
            exc.append(ln.strip())

    return {
        "Insurer": insurer,
        "PolicyType": ptype,
        "SumInsured": si,
        "Premium": prem,
        "Deductible": "",
        "Inclusions": inc[:5],
        "Exclusions": exc[:5],
    }


# ---------------------------------------------------------------------
# POLICY READER
# ---------------------------------------------------------------------
class PolicyReader:
    def __init__(self, pdf_bytes):
        self.raw_text = self._extract(pdf_bytes)

    def _extract(self, pdf_bytes):
        try:
            pages = []
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    pages.append(p.extract_text() or "")
            return "\n".join(pages)
        except:
            return ""

    def _chunks(self, text, size=3500, overlap=500):
        out = []
        i = 0
        while i < len(text):
            out.append(text[i:i+size])
            i += size - overlap
        return out or [text]

    def _merge(self, final, part):
        for k in ["Insurer", "PolicyType", "SumInsured", "Premium", "Deductible"]:
            v = part.get(k)
            if v and str(v).lower() not in ["", "unknown", "null"]:
                if not final.get(k) or (k in ["SumInsured", "Premium"] and isinstance(v, (int, float))):
                    final[k] = v
        final["Inclusions"] = list(set(final["Inclusions"] + part.get("Inclusions", [])))
        final["Exclusions"] = list(set(final["Exclusions"] + part.get("Exclusions", [])))
        return final

    def extract_fields(self):
        chunks = self._chunks(self.raw_text)
        final = {
            "Insurer": "Unknown",
            "PolicyType": "Unknown",
            "SumInsured": "",
            "Premium": "",
            "Deductible": "",
            "Inclusions": [],
            "Exclusions": [],
        }

        success = False

        for c in chunks:
            prompt = f"""
You are CredenceX. Output ONLY valid JSON.
SCHEMA: {{"Insurer": "str", "PolicyType": "Health|Life|Motor|Home|Unknown",
"SumInsured": "num/str", "Premium": "num/str", "Deductible": "str",
"Inclusions": [], "Exclusions": []}}

Extract from:
\"\"\"{c}\"\"\"
"""
            raw = call_llama(prompt, json_mode=True)
            parsed = try_parse_json(raw)
            if not parsed:
                cleaned = raw.replace("```json", "").replace("```", "")
                parsed = try_parse_json(cleaned)

            if parsed:
                final = self._merge(final, parsed)
                success = True

            time.sleep(0.5)

        if not success:
            final = regex_extract_fields(self.raw_text)

        final["SumInsured"] = normalize_number(final.get("SumInsured"))
        final["Premium"] = normalize_number(final.get("Premium"))
        return final


# ---------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------
def score_policy(policy, user, ptype):
    score = 0
    si = policy.get("SumInsured") or 0
    prem = policy.get("Premium") or 0

    score += min(si / 100000, 20)
    if prem > 0:
        score += max(0.5, 10 - (prem / 20000))

    if "%" in str(policy.get("Deductible", "")):
        score -= 2

    score -= len(policy.get("Exclusions", [])) * 1.25

    if policy.get("PolicyType", "").lower() == ptype.lower():
        score += 2

    return round(score, 2)


# ---------------------------------------------------------------------
# GAP ANALYSIS
# ---------------------------------------------------------------------
def detect_gaps(policy, ptype, user):
    gaps = []
    si = policy.get("SumInsured") or 0
    exc = policy.get("Exclusions", [])

    if ptype == "Health":
        rec = 500000 if user.get("age", 35) < 45 else 1000000
        if si < rec:
            gaps.append(f"Sum insured low ({si}), recommended {rec}.")

    if ptype == "Life":
        rec = int(user.get("annual_income", 0) * 20)
        if si < rec:
            gaps.append(f"Under-insured. Recommended {rec} (found {si}).")

    return gaps


# ---------------------------------------------------------------------
# CHART GENERATOR
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# NEW UNIVERSAL VISUALIZATION — SCATTER + RADAR (WORKS FOR 1–5 POLICIES)
# ---------------------------------------------------------------------
def create_chart_image(parsed, scores):
    labels = [f"P{i+1}" for i in range(len(parsed))]
    premiums = [p.get("Premium") or 0 for p in parsed]
    covers = [p.get("SumInsured") or 0 for p in parsed]
    inc_counts = [len(p.get("Inclusions") or []) for p in parsed]
    exc_counts = [len(p.get("Exclusions") or []) for p in parsed]

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(16, 5))

# ---------------- Bar: Premium -----------------
    ax1 = fig.add_subplot(1, 4, 1)
    bars1 = ax1.bar(labels, premiums, color="#ff5c5c", width=0.5) # Soft Red
    ax1.set_title("Premium", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Amount", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add numbers on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=9)

    # ---------------- Bar: Sum Insured ----------------
    ax2 = fig.add_subplot(1, 4, 2)
    bars2 = ax2.bar(labels, covers, color="#3b82f6", width=0.5) # Electric Blue
    ax2.set_title("Sum Insured", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add numbers on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=9)

    # ---------------- Bar: Scores -----------------
    ax3 = fig.add_subplot(1, 4, 3)
    bars3 = ax3.bar(labels, scores, color="#10b981", width=0.5) # Emerald Green
    ax3.set_title("AI Score", fontsize=11, fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    # Add numbers on top of bars
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom', ha='center', fontsize=9)

    # ---------------- Radar Chart -----------------
    ax4 = fig.add_subplot(1, 4, 4, polar=True)

    # Radar metrics
    metrics = ["Premium", "SumInsured", "Score", "Inclusions", "Exclusions"]
    N = len(metrics)

    # Normalize values so radar looks good
    r_prem = premiums[0] / max(premiums) if max(premiums) != 0 else 0
    r_cov = covers[0] / max(covers) if max(covers) != 0 else 0
    r_score = scores[0] / max(scores) if max(scores) != 0 else 0
    r_inc = inc_counts[0] / max(inc_counts) if max(inc_counts) != 0 else 0
    r_exc = exc_counts[0] / max(exc_counts) if max(exc_counts) != 0 else 0

    values = [r_prem, r_cov, r_score, r_inc, r_exc]
    values += values[:1]  # close loop

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax4.plot(angles, values, color="purple", linewidth=2)
    ax4.fill(angles, values, color="purple", alpha=0.3)
    ax4.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax4.set_title("Policy Radar Summary", pad=20)

    # ---------------- Save & Return -----------------
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf



# ---------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------
def generate_pdf(user, ptype, parsed, scores, best, gaps):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 40

    def write(t, space=14):
        nonlocal y
        c.drawString(40, y, str(t))
        y -= space
        if y < 80:
            c.showPage()
            y = h - 40

    c.setFont("Helvetica-Bold", 14)
    write("CredenceX – Insurance Report", 20)

    # User
    c.setFont("Helvetica-Bold", 11)
    write("User Profile:", 18)
    c.setFont("Helvetica", 10)
    for k, v in user.items():
        write(f"{k}: {v}")

    # chart
    try:
        chart_buf = create_chart_image(parsed, scores)
        img = ImageReader(chart_buf)
        c.drawImage(img, 40, y - 160, width=520, height=150)
        y -= 190
    except:
        write("[Chart failed]")

    # policies
    c.setFont("Helvetica-Bold", 11)
    write("Policies:", 18)
    c.setFont("Helvetica", 10)

    for i, p in enumerate(parsed):
        write(f"Policy {i+1}: {p.get('Insurer')} - {p.get('PolicyType')}")
        write(f"SumInsured: {p.get('SumInsured')} | Premium: {p.get('Premium')} | Score: {scores[i]}")
        if p.get("Inclusions"):
            write("Inclusions: " + ", ".join(p["Inclusions"][:3]))
        if p.get("Exclusions"):
            write("Exclusions: " + ", ".join(p["Exclusions"][:3]))
        write("")

    # recommendation
    c.setFont("Helvetica-Bold", 11)
    write("Recommendation:", 18)
    c.setFont("Helvetica", 10)
    write(f"Insurer: {best.get('Insurer')}")
    write(f"SumInsured: {best.get('SumInsured')} | Premium: {best.get('Premium')}")

    # gaps
    c.setFont("Helvetica-Bold", 11)
    write("Gap Analysis:", 18)
    c.setFont("Helvetica", 10)
    if gaps:
        for g in gaps:
            write(f"- {g}")
    else:
        write("No major gaps detected.")

    c.save()
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    insurance_type = request.form.get("insurance_type")
    user_data = json.loads(request.form.get("user"))

    files = request.files.getlist("pdfs[]")

    parsed = []
    ctx = ""

    for f in files:
        data = f.read()
        reader = PolicyReader(data)

        ctx += f"\n\n--- {f.filename} ---\n" + reader.raw_text
        parsed.append(reader.extract_fields())

    scores = [score_policy(p, user_data, insurance_type) for p in parsed]
    best = parsed[np.argmax(scores)]
    gaps = detect_gaps(best, insurance_type, user_data)

    # Chart for HTML
    chart_buf = create_chart_image(parsed, scores)
    chart_b64 = base64.b64encode(chart_buf.getvalue()).decode()

    # PDF
    pdf_buf = generate_pdf(user_data, insurance_type, parsed, scores, best, gaps)
    pdf_b64 = base64.b64encode(pdf_buf.getvalue()).decode()

    return jsonify({
        "parsed": parsed,
        "scores": scores,
        "best": best,
        "gaps": gaps,
        "chart": chart_b64,
        "pdf": pdf_b64,
        "context": ctx
    })


@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message")
    ctx = request.form.get("context", "")

    prompt = f"""
Answer ONLY using the text below.
If not found, say: "This information is not mentioned."

Document Context:
----------------
{ctx[:15000]}

User Question:
{message}

Answer:
"""
    ans = call_llama(prompt, json_mode=False)

    return jsonify({"answer": ans})


# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
