"""Extract full text of job_description.docx, paragraph by paragraph."""
import zipfile, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

p = Path(r"A:\side_hustle\IndiaRuns Hackathon") / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "job_description.docx"

def unescape(s):
    return (s.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
             .replace("&quot;",'"').replace("&apos;","'"))

with zipfile.ZipFile(p) as z:
    xml = z.read("word/document.xml").decode("utf-8")

paras = re.split(r"</w:p>", xml)
n = 0
for para in paras:
    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.DOTALL)
    line = unescape("".join(texts)).strip()
    if line:
        n += 1
        print(f"{n:>3}| {line}")

print("\n" + "="*60)
# scan for education-related keywords across the whole doc
full = unescape(" ".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.DOTALL))).lower()
for kw in ["education","degree","college","university","tier","iit","nit","gpa",
           "grade","academic","b.tech","m.tech","phd","institut","alma","graduat","cgpa","cs degree","computer science"]:
    c = full.count(kw)
    print(f"  '{kw}': {c} occurrence(s)")
