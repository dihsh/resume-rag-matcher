"""
test_pipeline.py - Automated Test Suite for Resume RAG & Job Matcher Engine
"""

import json
import unittest
from pathlib import Path
from resume_rag import ResumeRAG
from job_matcher import JobMatcher

RESUMES_DIR = Path("resumes")
JDS_DIR = Path("job_descriptions")


class TestResumeRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = ResumeRAG()
        cls.matcher = JobMatcher(rag=cls.rag)

    def test_01_resumes_dataset_exists(self):
        resumes = list(RESUMES_DIR.glob("*.*"))
        self.assertGreaterEqual(len(resumes), 30, f"Expected 30+ resumes, found {len(resumes)}")

    def test_02_jds_dataset_exists(self):
        jds = list(JDS_DIR.glob("*.json"))
        self.assertGreaterEqual(len(jds), 5, f"Expected 5+ job descriptions, found {len(jds)}")

    def test_03_vector_store_populated(self):
        count = self.rag.collection.count()
        self.assertGreaterEqual(count, 100, f"Expected 100+ chunks in ChromaDB, found {count}")

    def test_04_section_chunking(self):
        sample_file = RESUMES_DIR / "resume_01_john_doe.pdf"
        from fs_tools import read_file
        res = read_file(str(sample_file))
        self.assertEqual(res["status"], "success")
        chunks = self.rag.chunk_document_by_sections(res["content"])
        self.assertGreaterEqual(len(chunks), 3)
        sections = [c["section"] for c in chunks]
        self.assertTrue(any(s in ["summary", "skills", "experience", "education"] for s in sections))

    def test_05_metadata_extraction(self):
        sample_file = RESUMES_DIR / "resume_02_emily_chen.pdf"
        from fs_tools import read_file
        res = read_file(str(sample_file))
        meta = self.rag.extract_metadata("resume_02_emily_chen.pdf", res["content"])
        self.assertIn("Emily Chen", meta["candidate_name"])
        self.assertIn("Python", meta["skills"])
        self.assertIn("SQL", meta["skills"])
        self.assertGreaterEqual(meta["experience_years"], 4)

    def test_06_semantic_search(self):
        query = "Distributed systems, Python FastAPI and Redis caching"
        results = self.rag.query_semantic(query, n_results=5)
        self.assertEqual(len(results), 5)
        top_name = results[0]["metadata"]["candidate_name"]
        self.assertIn("John Doe", top_name)

    def test_07_job_matching_schema(self):
        jd_path = str(JDS_DIR / "jd_01_senior_python_backend.json")
        result = self.matcher.match_job(jd_path, top_k=5)
        
        self.assertIn("job_description", result)
        self.assertIn("top_matches", result)
        self.assertEqual(len(result["top_matches"]), 5)

        first = result["top_matches"][0]
        self.assertIn("candidate_name", first)
        self.assertIn("resume_path", first)
        self.assertIn("match_score", first)
        self.assertIn("matched_skills", first)
        self.assertIn("relevant_excerpts", first)
        self.assertIn("reasoning", first)
        self.assertIsInstance(first["match_score"], int)
        self.assertTrue(0 <= first["match_score"] <= 100)

    def test_08_hybrid_ranking_accuracy(self):
        jd_path = str(JDS_DIR / "jd_02_ml_nlp_engineer.json")
        result = self.matcher.match_job(jd_path, top_k=3)
        top_candidates = [m["candidate_name"] for m in result["top_matches"]]
        
        # Elena Rostova, Alex Kumar, or Priya Sharma should be in top 3
        expected_ml_experts = ["Elena Rostova", "Alex Kumar", "Priya Sharma"]
        matches_found = [c for c in top_candidates if any(exp in c for exp in expected_ml_experts)]
        self.assertGreaterEqual(len(matches_found), 2)


if __name__ == "__main__":
    unittest.main()
