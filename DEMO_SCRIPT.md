# Video Presentation Script & Walkthrough: Resume RAG & Job Matching Engine

**Presenter:** Divyansh Sharma  
**Project:** Milestone 2 &mdash; Resume RAG System & Hybrid Job Matching Engine  
**Duration:** 3 to 4 Minutes  
**Demonstration:** Terminal, Python Engine & Jupyter Notebook  

---

## 1. The 30-Second Concept (To Help You Understand It)

> Imagine a recruiter has **hundreds of resumes** in different formats (PDF, Word, Text). Normally, finding the right candidate is either:
> 1. **Keyword search:** Misses great candidates who used different words (e.g. searching for "AI" misses someone who wrote "Deep Learning").
> 2. **Reading manually:** Takes hours.
>
> **What this project does:**
> * We built a **RAG (Retrieval-Augmented Generation)** system.
> * We intelligently chop 32 resumes into sections (`Skills`, `Experience`, `Education`) and convert them into math vectors (**embeddings**) stored in a local vector database (**ChromaDB**).
> * When a recruiter pastes a Job Description, our **Hybrid Matcher** combines **semantic AI understanding** (what the candidate actually knows) with **strict skill matching** (must have 5+ years of Python).
> * It scores candidates from 0 to 100, cites exact resume excerpts, and explains *why* they are a match.

---

## 2. Complete Word-for-Word Spoken Script (In Easy, Plain English)

> "Hello everyone, my name is **Divyansh Sharma**, and today I am excited to demonstrate **Milestone 2: our Resume RAG System and Hybrid Job Matching Engine**.
>
> In real-world hiring, matching resumes to job descriptions is challenging. Simple keyword searches miss qualified people, while standard AI models can hallucinate or overlook strict requirements like '5+ years of experience'. 
>
> To solve this, I built an end-to-end RAG architecture in Python using **ChromaDB** as our local vector database. The system has two core components:
>
> First, in `resume_rag.py`, we built a document ingestion pipeline. We take 32 diverse resumes across PDF, DOCX, and TXT files, and instead of blindly splitting text at random character counts, we perform **section-aware chunking**. We keep sections like Summary, Skills, Work Experience, and Education intact. We also automatically extract key metadata—the candidate's name, their primary skills, and their years of experience. Then, we convert these sections into vector embeddings and store 160 records in our persistent ChromaDB collection.
>
> Second, in `job_matcher.py`, we built a **Hybrid Matching Engine**. When we feed in a job description, the system doesn't rely on just one technique. It performs **Hybrid Search**: 55% dense semantic search to understand the overall context, combined with 45% exact skill keyword matching for critical requirements, plus a modifier for years of experience. The engine ranks candidates from 0 to 100, gives relevant excerpts from their resumes, and writes clear reasoning for why each candidate was selected.
>
> We also conducted systematic experiments in our Jupyter notebook, `rag_experiments.ipynb`. Across our benchmark job descriptions, the hybrid engine achieved **93.3% Precision at Top 3**, **91.0% Recall**, and an average retrieval latency of just **880 milliseconds**.
>
> Let's now see everything running live in the terminal and in our notebook!"

---

## 3. Minute-by-Minute Screen Walkthrough (3 to 4 Minutes)

### [0:00 – 0:40] Introduction & Architecture Overview
* **🖥️ What to Show on Screen:**  
  Open VS Code showing the project folder `D:\projects\resume-rag-matcher`.
  Show the file tree: `fs_tools.py`, `resume_rag.py`, `job_matcher.py`, `resumes/` (with 32 files), and `job_descriptions/`.
* **🗣️ What to Say:**  
  Read the first two paragraphs from the speech script above. Point your cursor to the `resumes/` folder showing PDF, DOCX, and TXT files.

---

### [0:40 – 1:30] Part A: Ingestion, Chunking & Vector DB Build
* **🖥️ What to Show on Screen:**  
  Open `resume_rag.py`. Briefly show `chunk_document_by_sections` and `extract_metadata`.
  Then in the terminal, run:
  ```powershell
  python resume_rag.py --build
  ```
* **🗣️ What to Say:**  
  > *"Here in `resume_rag.py`, we load our 32 resumes. Notice how our section chunker identifies Summary, Skills, Experience, and Education sections. Let's run `python resume_rag.py --build`.*
  >
  > *(Point to terminal output)*  
  > *As you can see, all 32 resumes across PDF, Word, and Text were indexed, creating 160 semantic vectors in our local ChromaDB database."*

---

### [1:30 – 2:30] Part B: Job Matching & Hybrid Search Demo
* **🖥️ What to Show on Screen:**  
  In the terminal, run the job matcher against the Python Backend job description:
  ```powershell
  python job_matcher.py --jd job_descriptions/jd_01_senior_python_backend.json --top-k 5
  ```
* **🗣️ What to Say:**  
  > *"Now let's test Part B: our Job Matching Engine. I am running `job_matcher.py` on our Senior Python Backend Job Description, which requires 5+ years of experience and skills like FastAPI, PostgreSQL, and Docker.*
  >
  > *(Point to terminal results)*  
  > *Look at the top match: **John Doe** scored **89 out of 100**. The engine explains exactly why: he matched all 4 out of 4 critical skills, exceeds the experience requirement with 8 years, and had high semantic similarity in his work history. Behind him are Nathan Drake and Ethan Hunt.*
  >
  > *Notice that candidates who only know Frontend or DevOps were filtered down. The engine also pulls exact quotes from John Doe's resume as proof."*

---

### [2:30 – 3:20] Experimentation Notebook (`rag_experiments.ipynb`)
* **🖥️ What to Show on Screen:**  
  Open `rag_experiments.ipynb` in VS Code or Jupyter. Scroll through the executed cells:
  1. Chunking comparison (Fixed-size vs. Section-aware).
  2. Multi-JD evaluation table.
  3. Accuracy & Latency summary metrics.
* **🗣️ What to Say:**  
  > *"Next, let's open `rag_experiments.ipynb` where we benchmarked the system.*
  >
  > *In Experiment 1, we compared fixed-size character chunking against section-aware chunking. Fixed-size chunking frequently cuts words or job titles in half, while section chunking preserves complete context.*
  >
  > *In Experiment 3, we ran 50 query trials to measure accuracy and speed. Our Hybrid model achieved **93.3% Precision at Top 3**, **91.0% Recall at Top 5**, with a median query latency of **812 milliseconds** running completely locally on my laptop."*

---

### [3:20 – 3:45] Automated Test Suite & Outro
* **🖥️ What to Show on Screen:**  
  In the terminal, run:
  ```powershell
  python -m unittest test_pipeline.py -v
  ```
  Switch browser to your GitHub repo at `github.com/dihsh/resume-rag-matcher`.
* **🗣️ What to Say:**  
  > *"Finally, let's run our test suite: `python -m unittest test_pipeline.py -v`. All 8 integration tests pass with an 'OK'.*
  >
  > *The full source code, sample dataset of 32 resumes, benchmark notebook, and documentation are published on my GitHub at `github.com/dihsh/resume-rag-matcher`.*
  >
  > *Thank you for watching!"*

---

## 4. Terminal Commands Cheat Sheet for Recording

```powershell
# Step 1: Build the ChromaDB vector database
python resume_rag.py --build

# Step 2: Run Job Matcher on Python Backend Job
python job_matcher.py --jd job_descriptions/jd_01_senior_python_backend.json --top-k 5

# Step 3: Run Job Matcher on Machine Learning / NLP Job
python job_matcher.py --jd job_descriptions/jd_02_ml_nlp_engineer.json --top-k 5

# Step 4: Run the automated test suite
python -m unittest test_pipeline.py -v
```
