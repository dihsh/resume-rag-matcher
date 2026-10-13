"""
job_matcher.py - Semantic & Hybrid Job Matching Engine for Resumes

Part B of Assignment:
1. Accepts Job Description (JSON or raw text).
2. Performs Hybrid Search: Dense ChromaDB Vector Search + Sparse Keyword Skill Matching.
3. Filters by must-have requirements (e.g. "5+ years Python").
4. Scores matches on a 0-100 scale and ranks top-K candidates (K=10).
5. Provides match reasoning and relevant excerpts.
6. Outputs strict JSON schema requested by the assignment.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Use Windows system trust store
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from resume_rag import ResumeRAG, KNOWN_SKILLS

load_dotenv()


class JobMatcher:
    def __init__(self, rag: Optional[ResumeRAG] = None):
        self.rag = rag or ResumeRAG()

    def parse_job_description(self, jd_input: str) -> Dict[str, Any]:
        """
        Parse JD from file path (JSON/TXT) or raw text string.
        Extracts title, description, min_experience_years, must_have_skills, nice_to_have_skills.
        """
        path = Path(jd_input)
        if path.exists() and path.is_file():
            raw_text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw_text)
                return {
                    "id": data.get("id", path.stem),
                    "title": data.get("title", path.stem.replace("_", " ").title()),
                    "description": data.get("description", raw_text),
                    "min_experience_years": data.get("min_experience_years", 0),
                    "must_have_skills": data.get("must_have_skills", []),
                    "nice_to_have_skills": data.get("nice_to_have_skills", []),
                }
            except json.JSONDecodeError:
                # Text file JD
                return self._extract_jd_from_text(raw_text, title=path.stem)
        else:
            return self._extract_jd_from_text(jd_input, title="Job Opportunity")

    def _extract_jd_from_text(self, text: str, title: str = "Job Opening") -> Dict[str, Any]:
        """Extract requirements from unformatted text using regex."""
        # Find years of experience
        exp_match = re.search(r"(\d+)\+?\s*years?(?:\s+of)?\s+experience", text, re.IGNORECASE)
        min_years = int(exp_match.group(1)) if exp_match else 0

        # Match technical skills
        text_lower = text.lower()
        found_skills = []
        for s in KNOWN_SKILLS:
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower):
                found_skills.append(s)

        # First 4 as must-haves, rest as nice-to-have
        must_haves = found_skills[:4] if len(found_skills) >= 4 else found_skills
        nice_to_haves = found_skills[4:] if len(found_skills) > 4 else []

        return {
            "id": "custom_jd",
            "title": title,
            "description": text,
            "min_experience_years": min_years,
            "must_have_skills": must_haves,
            "nice_to_have_skills": nice_to_haves,
        }

    def match_job(
        self,
        jd_input: str,
        top_k: int = 10,
        strict_filter: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Hybrid Search (Semantic Vector Search + Keyword Matching) and ranking.
        """
        jd_meta = self.parse_job_description(jd_input)
        jd_text = jd_meta["description"]
        min_exp = jd_meta.get("min_experience_years", 0)
        must_haves = [s.strip() for s in jd_meta.get("must_have_skills", [])]
        nice_to_haves = [s.strip() for s in jd_meta.get("nice_to_have_skills", [])]

        # Query vector store for top chunks (over-fetch to aggregate by candidate)
        raw_chunks = self.rag.query_semantic(jd_text, n_results=60)

        # Aggregate chunk matches by candidate
        candidate_matches: Dict[str, Dict[str, Any]] = {}

        for chunk in raw_chunks:
            meta = chunk["metadata"]
            cand_name = meta["candidate_name"]
            sim = chunk["similarity"]
            resume_path = meta.get("resume_path", "")
            cand_exp = int(meta.get("experience_years", 0))
            cand_skills_str = meta.get("skills", "")
            cand_skills = [s.strip() for s in cand_skills_str.split(",") if s.strip()]

            if cand_name not in candidate_matches:
                candidate_matches[cand_name] = {
                    "candidate_name": cand_name,
                    "resume_path": resume_path,
                    "experience_years": cand_exp,
                    "all_skills": cand_skills,
                    "best_similarity": sim,
                    "matched_chunks": [],
                    "sections_matched": set(),
                }

            cm = candidate_matches[cand_name]
            if sim > cm["best_similarity"]:
                cm["best_similarity"] = sim
            cm["sections_matched"].add(meta.get("section", "general"))
            cm["matched_chunks"].append({
                "section": meta.get("section", "general"),
                "similarity": sim,
                "text": chunk["document"],
            })

        # Score and rank candidates
        ranked_candidates = []

        for cand_name, data in candidate_matches.items():
            cand_exp = data["experience_years"]
            cand_skills_lower = [s.lower() for s in data["all_skills"]]

            # Check must-have skills
            matched_must_haves = [s for s in must_haves if s.lower() in cand_skills_lower]
            matched_nice_to_haves = [s for s in nice_to_haves if s.lower() in cand_skills_lower]
            all_matched_skills = list(dict.fromkeys(matched_must_haves + matched_nice_to_haves))

            # Strict filter option
            if strict_filter:
                if min_exp > 0 and cand_exp < min_exp:
                    continue
                if must_haves and len(matched_must_haves) < len(must_haves):
                    continue

            # 1. Semantic Score (0 to 100)
            semantic_score = data["best_similarity"] * 100.0

            # 2. Skill Overlap Score (0 to 100)
            if must_haves:
                must_have_ratio = len(matched_must_haves) / len(must_haves)
            else:
                must_have_ratio = 1.0

            if nice_to_haves:
                nice_have_ratio = len(matched_nice_to_haves) / len(nice_to_haves)
            else:
                nice_have_ratio = 0.5

            skill_score = (must_have_ratio * 70.0) + (nice_have_ratio * 30.0)

            # 3. Experience Alignment (multiplier/penalty)
            exp_modifier = 1.0
            if min_exp > 0:
                if cand_exp >= min_exp:
                    exp_modifier = 1.05  # bonus for meeting/exceeding requirement
                else:
                    deficit = min_exp - cand_exp
                    exp_modifier = max(0.65, 1.0 - (deficit * 0.10))  # gentle penalty

            # 4. Final Hybrid Score (55% Semantic + 45% Skill Overlap) * Exp Modifier
            hybrid_score = ((semantic_score * 0.55) + (skill_score * 0.45)) * exp_modifier
            final_score = int(round(max(10, min(99, hybrid_score))))

            # Extract top 2 excerpts
            sorted_chunks = sorted(data["matched_chunks"], key=lambda x: x["similarity"], reverse=True)
            excerpts = []
            for ch in sorted_chunks[:2]:
                lines = [l.strip() for l in ch["text"].splitlines() if l.strip()]
                # Grab content lines omitting headers
                body_lines = [l for l in lines if not any(l.startswith(prefix) for prefix in ["Candidate:", "Section:", "Experience:", "Key Skills:"])]
                if body_lines:
                    snippet = " ".join(body_lines[:3])
                    if len(snippet) > 220:
                        snippet = snippet[:220] + "..."
                    excerpts.append(snippet)

            # Generate natural language reasoning
            sections_str = ", ".join(sorted(list(data["sections_matched"])))
            reasoning_parts = []
            
            if matched_must_haves:
                reasoning_parts.append(f"Matched {len(matched_must_haves)}/{len(must_haves)} critical skills ({', '.join(matched_must_haves)})")
            
            if min_exp > 0:
                if cand_exp >= min_exp:
                    reasoning_parts.append(f"meets experience requirement with {cand_exp} years (required {min_exp}+ years)")
                else:
                    reasoning_parts.append(f"has {cand_exp} years experience (target: {min_exp}+ years)")

            reasoning_parts.append(f"high semantic alignment in sections [{sections_str}]")
            reasoning_str = ". ".join(reasoning_parts).capitalize() + "."

            ranked_candidates.append({
                "candidate_name": cand_name,
                "resume_path": data["resume_path"],
                "match_score": final_score,
                "matched_skills": all_matched_skills,
                "relevant_excerpts": excerpts,
                "reasoning": reasoning_str,
                "_sort_key": final_score,
            })

        # Sort by match score descending
        ranked_candidates = sorted(ranked_candidates, key=lambda x: x["_sort_key"], reverse=True)

        # Remove internal sort key
        top_matches = []
        for c in ranked_candidates[:top_k]:
            c_clean = {k: v for k, v in c.items() if not k.startswith("_")}
            top_matches.append(c_clean)

        return {
            "job_description": jd_text,
            "top_matches": top_matches,
        }


def main():
    parser = argparse.ArgumentParser(description="Job Matching Engine using Hybrid RAG Search")
    parser.add_argument("--jd", type=str, default="job_descriptions/jd_01_senior_python_backend.json", help="Path to JD file or raw text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to return (default: 10)")
    parser.add_argument("--strict", action="store_true", help="Enforce strict filtering on experience and must-haves")
    parser.add_argument("--output", type=str, help="Optional output JSON file path")

    args = parser.parse_args()
    matcher = JobMatcher()

    print(f"\n=======================================================")
    print(f"  JOB MATCHING & HYBRID RETRIEVAL ENGINE")
    print(f"=======================================================")
    print(f"Job Description Input: {args.jd}")
    print(f"Top-K: {args.top_k} | Strict Filter: {args.strict}\n")

    result = matcher.match_job(args.jd, top_k=args.top_k, strict_filter=args.strict)

    # Pretty print summary
    print(f"Found {len(result['top_matches'])} top matches:")
    for idx, m in enumerate(result["top_matches"], start=1):
        print(f"\n[{idx}] {m['candidate_name']} &mdash; Match Score: {m['match_score']}/100")
        print(f"    Path    : {m['resume_path']}")
        print(f"    Skills  : {', '.join(m['matched_skills'])}")
        print(f"    Reason  : {m['reasoning']}")
        if m["relevant_excerpts"]:
            print(f"    Excerpt : \"{m['relevant_excerpts'][0][:110]}...\"")

    # Save to file if requested
    if args.output:
        out_p = Path(args.output)
        out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[+] Saved detailed JSON results to: {out_p.resolve()}")

    # Output JSON string for programmatic consumption
    json_output = json.dumps(result, indent=2)
    return json_output


if __name__ == "__main__":
    main()
