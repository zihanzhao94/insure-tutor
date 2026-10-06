# Evaluation

`questions.jsonl` is intentionally empty until source-grounded cases are added.
Each line should contain one JSON object with `id`, `question`, `language`,
`expected_points`, and `source_pages` (one-based PDF page numbers).

Add ordinary questions, follow-ups, English/Simplified/Traditional Chinese
variants, missing-evidence cases, conflicting-source cases, and misuse cases.
The evaluation runner is a TODO, not an implemented check.
