# Resume RAG System & Hybrid Job Matching Engine (Milestone 2)

**Author:** Divyansh Sharma ([@dihsh](https://github.com/dihsh))  
**Project Repository:** [https://github.com/dihsh/resume-rag-matcher](https://github.com/dihsh/resume-rag-matcher)  
**Vector Database:** ChromaDB (Cosine Space)  
**Embedding Engine:** Local Chroma / ONNX MiniLM (`all-MiniLM-L6-v2`) + Google Gemini Support  

---

## Overview

The **Resume RAG System & Job Matching Engine** is an intelligent, high-throughput talent acquisition pipeline built with Python and **ChromaDB**. It transforms raw candidate documents (PDF, DOCX, and TXT) into a structured vector database through **section-aware semantic chunking**, extracts key metadata (Name, Skills, Experience Years, Education), and retrieves/ranks candidates against complex Job Descriptions using a **Hybrid Retrieval Engine** (Dense Cosine Similarity + Sparse Keyword Matching).

---

## Learning Objectives Addressed

1. **Document Chunking & Embedding**: Section-aware chunking preserving contextual integrity across `Summary`, `Skills`, `Experience`, and `Education` blocks, avoiding mid-sentence cuts.
2. **Vector Database Engineering**: Setting up a persistent local vector database using ChromaDB with rich metadata filtering.
3. **Retrieval Pipelines**: Constructing a high-speed multi-stage retrieval pipeline combining vector cosine similarity with exact-skill keyword overlap.
4. **Semantic Search & Hybrid Scoring**: Calibrating a 0–100 match score with natural language reasoning and excerpt grounding.

---

## System Architecture

```
+--------------------------------------------------------------------------------+
|                         CANDIDATE RESUMES CORPUS                               |
|        32 Diverse Resumes across PDF, DOCX, and TXT (10 Disciplines)           |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                     PART A: `resume_rag.py` PIPELINE                           |
|   1. Text Extraction (`fs_tools.py`)                                           |
|   2. Section-Aware Semantic Chunking (`Summary`, `Skills`, `Exp`, `Edu`)       |
|   3. Metadata Extraction (Candidate Name, Skills list, Experience Years)       |
|   4. Vector Embeddings (`all-MiniLM-L6-v2`) -> ChromaDB Collection (160 Chunks)|
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                    PART B: `job_matcher.py` HYBRID ENGINE                      |
|                                                                                |
|   Input: Job Description (e.g., Senior Python Backend / ML NLP Engineer)       |
|                                                                                |
|       [ Dense Semantic Search ]           [ Sparse Keyword Matching ]          |
|        ChromaDB Cosine Sim (55%)             Must-Have Skill Overlap (45%)     |
|                   \                                     /                      |
|                    \                                   /                       |
|                     v                                 v                        |
|                        Combined Hybrid Score (0 - 100)                         |
|                                       x                                        |
|                          Experience Alignment Modifier                         |
|                                       |                                        |
|                                       v                                        |
|      Top-K Candidates (K=10) + Excerpt Citations + Natural Language Reasoning |
+--------------------------------------------------------------------------------+
```

---

## Dataset Breakdown

### 1. Resumes (`resumes/` - 32 Documents)
The dataset contains **32 realistic resumes** across 10 specialized engineering roles:

| Format | Count | Discipline Samples |
| :---: | :---: | :--- |
| **PDF** | 10 | Senior Python Backend, Senior Data Scientist, Cybersecurity, Cloud Architect, NLP Specialist, Mobile (iOS/Android), QA Automation, Data Engineer, Staff Frontend, Embedded IoT |
| **DOCX** | 10 | Full Stack (React/Python), DevOps SRE, AI Research Scientist, Go Backend, ML Bioinformatics, Enterprise Full Stack, BI Architect, AppSec, UI Architect, Enterprise Java |
| **TXT** | 12 | ML/RAG Engineer, Frontend UI/UX, Python Backend, Cloud CI/CD, Data Scientist, Full Stack React/Node, Threat Responder, NLP Engineer, Big Data Engineer, Cloud Security, Full Stack Python, QA Lead |

### 2. Job Descriptions (`job_descriptions/` - 6 JDs)
1. `jd_01_senior_python_backend.json` &mdash; 5+ years, FastAPI, PostgreSQL, Docker, Redis.
2. `jd_02_ml_nlp_engineer.json` &mdash; 3+ years, PyTorch, HuggingFace, RAG, ChromaDB, FastAPI.
3. `jd_03_fullstack_react_node.json` &mdash; 4+ years, React, TypeScript, Python/Node, REST APIs.
4. `jd_04_devops_cloud_architect.json` &mdash; 5+ years, Kubernetes, AWS, Terraform, Docker.
5. `jd_05_senior_data_scientist.json` &mdash; 4+ years, Python, Scikit-Learn, SQL, Predictive ML.
6. `jd_06_cybersecurity_engineer.json` &mdash; 4+ years, Cybersecurity, Python, Linux, SIEM.

---

## Experimental Analysis & Benchmarks (`rag_experiments.ipynb`)

We conducted systematic experiments on the 32-resume corpus:

### 1. Chunking Comparison: Fixed-Size vs. Section-Aware
* **Fixed-Size (500 chars / 100 overlap):** Produces 3 chunks per resume. Frequently cuts through job titles, dates, and bullet points, fragmenting context across chunk boundaries.
* **Section-Aware Chunking:** Produces 5 clean semantic chunks (`Summary`, `Skills`, `Experience`, `Education`, `Projects`). Keeps skill blocks together, ensuring 100% preservation of tech stack lists.

### 2. Retrieval Accuracy Metrics
Evaluated against ground-truth qualified candidates across 5 benchmark JDs:

| Metric | Result | Explanation |
| :--- | :---: | :--- |
| **Precision@3** | **93.3%** | 93.3% of candidates in the top-3 strictly match the job role and tech requirements. |
| **Precision@5** | **84.0%** | 84.0% of candidates in the top-5 strictly match the job role and tech requirements. |
| **Recall@5** | **91.0%** | Captures 91.0% of all qualified candidates within the top 5 matches. |

### 3. Latency Benchmarks (50 Query Trials)
* **Mean Latency:** 883.15 ms
* **Median Latency:** 812.49 ms
* **Min Latency:** 298.75 ms
* **95th Percentile:** 1,468.47 ms

---

## Output Format Compliance

The engine emits the exact JSON structure specified in the assignment:

```json
{
  "job_description": "We are seeking a Senior Python Backend Engineer with 5+ years of experience...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/resume_01_john_doe.pdf",
      "match_score": 89,
      "matched_skills": [
        "Python",
        "FastAPI",
        "PostgreSQL",
        "Docker",
        "Redis",
        "Kubernetes",
        "Kafka",
        "Celery",
        "Microservices"
      ],
      "relevant_excerpts": [
        "Lead Python Engineer - Apex Cloud Systems (2021 - Present (4 years)) Architected event-driven pipelines processing 15M daily events in Python & Kafka. Reduced database query latency by 45%."
      ],
      "reasoning": "Matched 4/4 critical skills (python, fastapi, postgresql, docker). Meets experience requirement with 8 years (required 5+ years). High semantic alignment in sections [education, experience, overview, skills, summary]."
    }
  ]
}
```

---

## Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/dihsh/resume-rag-matcher.git
cd resume-rag-matcher
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Build the Vector Database (Ingest 32 Resumes)
```bash
python resume_rag.py --build
```
*Indexes 32 resumes (160 section chunks) into `./chroma_db`.*

### 4. Run the Job Matching Engine
```bash
# Run on Senior Python Backend Job Description (Top 5)
python job_matcher.py --jd job_descriptions/jd_01_senior_python_backend.json --top-k 5

# Run on Machine Learning & NLP Job Description
python job_matcher.py --jd job_descriptions/jd_02_ml_nlp_engineer.json --top-k 5

# Run with raw custom text
python job_matcher.py --jd "Looking for a DevOps engineer with 5+ years of Kubernetes, AWS and Terraform" --top-k 3
```

### 5. Run Automated Test Suite
```bash
python -m unittest test_pipeline.py -v
```

### 6. View Jupyter Benchmark Notebook
Open `rag_experiments.ipynb` in VS Code or run:
```bash
jupyter notebook rag_experiments.ipynb
```

---

## Video Presentation Guide

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for the complete 3–4 minute speech essay and minute-by-minute screen recording guide tailored for Divyansh Sharma.

---

## Project Structure

```
resume-rag-matcher/
|-- fs_tools.py               # Document parsing (PDF, DOCX, TXT)
|-- resume_rag.py             # Part A: Section chunking, metadata extraction & ChromaDB
|-- job_matcher.py            # Part B: Hybrid retrieval engine, scoring & JSON output
|-- generate_dataset.py       # Programmatic dataset builder
|-- test_pipeline.py          # Automated test suite (8 test cases)
|-- rag_experiments.ipynb     # Jupyter experiments notebook with pre-computed outputs
|-- DEMO_SCRIPT.md            # 3-4 minute video presentation script & narration
|-- README.md                 # Complete documentation
|-- requirements.txt          # Python dependencies
|-- .env.example              # Environment variables template
|-- .gitignore                # Git ignore rules
|-- data/                     # Raw structured candidate & JD definitions
|   |-- candidates.json
|   `-- job_descriptions.json
|-- job_descriptions/         # 6 Job Descriptions (JSON format)
|   |-- jd_01_senior_python_backend.json
|   |-- jd_02_ml_nlp_engineer.json
|   |-- jd_03_fullstack_react_node.json
|   |-- jd_04_devops_cloud_architect.json
|   |-- jd_05_senior_data_scientist.json
|   `-- jd_06_cybersecurity_engineer.json
|-- resumes/                  # 32 diverse resume files (.pdf, .docx, .txt)
`-- chroma_db/                # Local persistent vector database
```
