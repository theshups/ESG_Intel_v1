---
title: ESG Document Intelligence
emoji: 📋
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
---

# ESG Document Classification & Anonymization Microservice

Upload a BRSR or Sustainability Report PDF. The pipeline:
1. **Ingests** PDF or TXT files
2. **Anonymizes** PII (person names, orgs, financials, CIN, PAN, email, phone, address)
3. **Classifies** the document (SEBI BRSR / Sustainability Report / Invalid)
4. **Scores** ESG performance across 16 sub-metrics (E×40% + S×35% + G×25%)
5. **Downloads** a full PDF report with charts and written analysis

## Run locally
```bash
pip install -r requirements.txt
python -m src.components.model_trainer   # train once
uvicorn main:app --reload --port 7860
```
Open http://localhost:7860
