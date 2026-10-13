"""
resume_rag.py - Document Processing Pipeline and Vector Database Setup for Resumes

Part A of Assignment:
1. Loads resumes using file system tools (PDF, DOCX, TXT).
2. Performs intelligent section-aware chunking (preserving Summary, Skills, Experience, Education).
3. Extracts candidate metadata: Name, Skills, Experience Years, Education.
4. Generates embeddings and stores chunks in a local persistent ChromaDB collection with filterable metadata.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Use Windows system trust store to prevent SSL certificate verification errors
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import chromadb
from chromadb.config import Settings

# Import file reading tools from Milestone 1
from fs_tools import read_file, list_files

load_dotenv()

CHROMA_PATH = Path("./chroma_db")
COLLECTION_NAME = "resume_sections"

KNOWN_SKILLS = [
    "Python", "Java", "Go", "Golang", "C++", "C", "Rust", "Swift", "Kotlin",
    "JavaScript", "TypeScript", "React", "Node.js", "Express", "FastAPI", "Django", "Flask",
    "Next.js", "Vue.js", "HTML5", "CSS3", "TailwindCSS", "Redux", "GraphQL",
    "PyTorch", "TensorFlow", "HuggingFace", "Transformers", "LangChain", "Vector Databases",
    "ChromaDB", "Scikit-Learn", "Machine Learning", "Deep Learning", "NLP",
    "Kubernetes", "Docker", "Terraform", "AWS", "GCP", "Azure", "CI/CD", "Helm",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Celery", "Snowflake", "BigQuery",
    "PySpark", "Apache Airflow", "dbt", "SQL", "Tableau", "PowerBI",
    "Cybersecurity", "SIEM (Splunk, Elastic)", "OWASP", "SOC 2", "Wireshark", "Penetration Testing",
    "QA Automation", "Selenium", "Playwright", "Cypress", "Jest", "PyTest",
    "Microservices", "REST APIs", "gRPC", "Linux", "Bash", "Git"
]

SECTION_HEADERS = {
    "summary": [r"professional\s+summary", r"summary", r"profile", r"about\s+me"],
    "skills": [r"technical\s+skills", r"skills", r"core\s+competencies", r"technologies"],
    "experience": [r"work\s+experience", r"experience", r"employment\s+history", r"professional\s+experience"],
    "education": [r"education", r"academic\s+background"],
    "projects": [r"key\s+projects", r"projects", r"notable\s+projects"],
}


class ResumeRAG:
    def __init__(self, persist_dir: str = str(CHROMA_PATH)):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Default Chroma embedding function uses all-MiniLM-L6-v2 locally via ONNX
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def extract_metadata(self, filename: str, content: str) -> Dict[str, Any]:
        """
        Extract structured metadata from resume content:
        Name, Skills, Experience Years, and Education.
        """
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        # 1. Candidate Name: from first non-header line or filename
        candidate_name = ""
        for line in lines[:5]:
            clean = re.sub(r"[=\-_*#]", "", line).strip()
            if clean and len(clean.split()) <= 4 and not any(kw in clean.lower() for kw in ["resume", "curriculum", "email", "phone"]):
                candidate_name = clean
                break
        
        if not candidate_name:
            # Fallback to filename formatting: e.g. resume_01_john_doe.pdf -> John Doe
            stem = Path(filename).stem
            parts = [p for p in stem.split("_") if not p.isdigit() and p.lower() != "resume"]
            candidate_name = " ".join(parts).title()

        # 2. Skills Extraction
        matched_skills = []
        content_lower = content.lower()
        for skill in KNOWN_SKILLS:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, content_lower):
                matched_skills.append(skill)

        # 3. Experience Years Extraction
        experience_years = 0
        exp_matches = re.findall(r"(\d+)\+?\s*years?(?:\s+of)?\s+experience", content_lower)
        if exp_matches:
            experience_years = max(int(m) for m in exp_matches)
        else:
            role_matches = re.findall(r"\((\d+)\s*years?\)", content_lower)
            if role_matches:
                experience_years = sum(int(m) for m in role_matches)
            else:
                years = [int(y) for y in re.findall(r"\b(20\d{2})\b", content)]
                if len(years) >= 2:
                    experience_years = max(years) - min(years)

        experience_years = max(1, min(experience_years, 20))

        # 4. Education Extraction
        education_str = "Bachelor's Degree in Computer Science"
        edu_match = re.search(r"(?:education|academic background)[:\s]+(.*?)(?:\n\n|\n[A-Z\s]{4,}:|$)", content, re.IGNORECASE | re.DOTALL)
        if edu_match:
            edu_snippet = edu_match.group(1).strip()
            first_line = edu_snippet.splitlines()[0].strip()
            if first_line:
                education_str = first_line[:120]

        return {
            "candidate_name": candidate_name,
            "skills": matched_skills,
            "experience_years": experience_years,
            "education": education_str,
        }

    def chunk_document_by_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Intelligently split a resume into meaningful semantic sections
        (Summary, Skills, Experience, Education, Projects).
        """
        lines = content.splitlines()
        sections: List[Dict[str, str]] = []
        current_section = "overview"
        current_lines = []

        header_patterns = []
        for sec, patterns in SECTION_HEADERS.items():
            for p in patterns:
                header_patterns.append((sec, re.compile(rf"^\s*(?:[#*=-]*\s*)?({p})\s*[:\-_=]*\s*$", re.IGNORECASE)))

        for line in lines:
            line_str = line.strip()
            detected_sec = None
            
            for sec, regex in header_patterns:
                if regex.match(line_str):
                    detected_sec = sec
                    break
            
            if detected_sec:
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        sections.append({
                            "section": current_section,
                            "content": text_block
                        })
                current_section = detected_sec
                current_lines = []
            else:
                current_lines.append(line)

        # Append final section
        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                sections.append({
                    "section": current_section,
                    "content": text_block
                })

        if len(sections) <= 1:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            sections = [{"section": f"section_{i+1}", "content": p} for i, p in enumerate(paragraphs)]

        return sections

    def process_and_index_resume(self, filepath: str) -> int:
        """
        Process a single resume: parse, chunk, extract metadata, and index in ChromaDB.
        """
        read_res = read_file(filepath)
        if read_res["status"] != "success":
            print(f"[ERROR] Failed reading {filepath}: {read_res.get('error')}")
            return 0

        content = read_res["content"]
        filename = Path(filepath).name
        metadata = self.extract_metadata(filename, content)
        chunks = self.chunk_document_by_sections(content)

        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{chunk['section']}_{idx}"
            
            enriched_text = (
                f"Candidate: {metadata['candidate_name']}\n"
                f"Section: {chunk['section'].upper()}\n"
                f"Experience: {metadata['experience_years']} years\n"
                f"Key Skills: {', '.join(metadata['skills'][:10])}\n\n"
                f"{chunk['content']}"
            )

            ids.append(chunk_id)
            documents.append(enriched_text)
            metadatas.append({
                "candidate_name": metadata["candidate_name"],
                "resume_path": filepath,
                "filename": filename,
                "section": chunk["section"],
                "experience_years": int(metadata["experience_years"]),
                "skills": ", ".join(metadata["skills"]),
                "education": metadata["education"],
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        return len(ids)

    def build_vector_store(self, resumes_dir: str = "resumes") -> Dict[str, Any]:
        """
        Iterate over all resumes in directory, chunk and index them into ChromaDB.
        """
        files = list_files(resumes_dir)
        total_chunks = 0
        indexed_resumes = 0

        print(f"\n=======================================================")
        print(f"  BUILDING RESUME RAG VECTOR DATABASE (ChromaDB)")
        print(f"=======================================================")
        print(f"Source Directory: {resumes_dir} ({len(files)} files discovered)")

        for file_info in files:
            path = file_info["path"]
            name = file_info["name"]
            num_chunks = self.process_and_index_resume(path)
            if num_chunks > 0:
                indexed_resumes += 1
                total_chunks += num_chunks
                print(f"  [+] Indexed {name:30} -> {num_chunks} chunks")

        total_in_db = self.collection.count()
        print(f"-------------------------------------------------------")
        print(f"Successfully indexed {indexed_resumes} resumes ({total_chunks} new chunks).")
        print(f"Total vector records in ChromaDB collection: {total_in_db}")
        print(f"Database persisted at: {self.persist_dir.resolve()}\n")

        return {
            "indexed_resumes": indexed_resumes,
            "total_chunks": total_chunks,
            "collection_count": total_in_db,
        }

    def query_semantic(self, query_text: str, n_results: int = 10, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform dense semantic vector retrieval against ChromaDB.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.5
                similarity = max(0.0, min(1.0, 1.0 - distance))

                matches.append({
                    "id": doc_id,
                    "document": doc_text,
                    "metadata": meta,
                    "similarity": similarity,
                })

        return matches


def main():
    parser = argparse.ArgumentParser(description="Resume RAG Ingestion & Vector DB Setup")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the ChromaDB vector database from resumes/")
    parser.add_argument("--resumes-dir", default="resumes", help="Path to resumes folder")
    parser.add_argument("--query", type=str, help="Test query against the vector store")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()
    rag = ResumeRAG()

    if args.build or not args.query:
        rag.build_vector_store(args.resumes_dir)

    if args.query:
        print(f"\n[QUERY]: '{args.query}' (Top {args.top_k})")
        results = rag.query_semantic(args.query, n_results=args.top_k)
        for idx, r in enumerate(results, start=1):
            meta = r["metadata"]
            print(f"\nMatch #{idx} (Similarity: {r['similarity']:.3f}):")
            print(f"  Candidate : {meta['candidate_name']} ({meta['filename']})")
            print(f"  Section   : {meta['section']} | Experience: {meta['experience_years']} years")
            print(f"  Snippet   : {r['document'].splitlines()[-1][:120]}...")


if __name__ == "__main__":
    main()
