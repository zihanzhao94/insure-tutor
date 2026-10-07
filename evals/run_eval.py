"""Source-grounded API checks. This is a smoke evaluation, not a semantic grader."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import httpx

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--ids", nargs="*", help="Run only these case IDs")
    args = parser.parse_args()
    cases = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text().splitlines() if line.strip()]
    if args.ids:
        unknown = set(args.ids) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.ids]
    results = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=120) as client:
        for case in cases:
            request = {"message": case["question"], "language": case["language"]}
            try:
                if case.get("prerequisite_question"):
                    prerequisite = client.post("/api/chat", json={"message": case["prerequisite_question"], "language": case["language"]})
                    prerequisite.raise_for_status()
                    request["session_id"] = prerequisite.json()["session_id"]
                response = client.post("/api/chat", json=request)
                response.raise_for_status()
                body = response.json()
                pages = sorted({citation["pdf_page"] for citation in body["citations"]})
                missing = [point for point in case["expected_points"]
                           if not any(value.lower() in body["answer"].lower()
                                      for value in (point if isinstance(point, list) else [point]))]
                checks = {"status": body["status"] in case["statuses"],
                          "source_page": not case["source_pages"] or bool(set(pages) & set(case["source_pages"])),
                          "expected_points": not missing,
                          "refusal_has_no_citations": body["status"] not in {"blocked", "out_of_scope", "insufficient_evidence"} or not body["citations"]}
                result = {"id": case["id"], "passed": all(checks.values()), "checks": checks,
                          "source_pages": pages, "missing_points": missing, "response": body}
            except (httpx.HTTPError, ValueError, KeyError) as error:
                result = {"id": case["id"], "passed": False, "error": str(error)}
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']}", flush=True)
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(result["passed"] for result in results)
    print(f"{passed}/{len(results)} checks passed. Report: {path}")
    print("Manually review factual conditions, entailment, dates, translations, and source-table fidelity.")
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
