from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.dataset.cleaner import clean_dataset
from src.dataset.formatter import sort_dataset, to_instruction_schema


DatasetItem = Dict[str, str]


@dataclass(frozen=True)
class TopicTemplate:
	topic: str
	prompts: Sequence[str]
	response_points: Sequence[Sequence[str]]


TEMPLATES: Sequence[TopicTemplate] = [
	TopicTemplate(
		topic="Python Programming",
		prompts=[
			"Explain the difference between lists and tuples in Python.",
			"How does list comprehension improve readability in Python?",
			"What is the purpose of Python virtual environments?",
			"How does exception handling work in Python with try and except?",
			"When should you use a dictionary instead of a list in Python?",
		],
		response_points=[
			[
				"Lists are mutable, while tuples are immutable.",
				"Tuples are useful for fixed collections and can be slightly faster.",
				"Lists are better when values need updates, inserts, or deletions.",
			],
			[
				"List comprehension combines iteration and transformation in one expression.",
				"It reduces boilerplate loops and keeps intent close to the data operation.",
				"Complex logic should still be moved to normal loops for clarity.",
			],
			[
				"Virtual environments isolate project dependencies from the system Python.",
				"They prevent version conflicts across projects.",
				"This makes builds and collaboration more reproducible.",
			],
			[
				"try runs code that may fail, and except handles specific errors.",
				"This avoids abrupt crashes and allows fallback behavior.",
				"Use specific exception types so debugging remains clear.",
			],
			[
				"Dictionaries provide key-based lookups with average constant-time access.",
				"Lists are better for ordered sequential data.",
				"Choose dictionaries when semantic keys matter.",
			],
		],
	),
	TopicTemplate(
		topic="Data Structures and Algorithms",
		prompts=[
			"Explain binary search in simple terms.",
			"What is the difference between stack and queue?",
			"How does a hash table handle collisions?",
			"When should we use BFS instead of DFS in graph traversal?",
			"What is the time complexity of merge sort and why?",
		],
		response_points=[
			[
				"Binary search repeatedly halves a sorted array to locate a value.",
				"It compares the middle element and discards one half each step.",
				"Its time complexity is logarithmic, making it efficient on large sorted data.",
			],
			[
				"A stack follows last-in, first-out behavior.",
				"A queue follows first-in, first-out behavior.",
				"Stacks are common in recursion; queues are common in scheduling.",
			],
			[
				"Collisions occur when multiple keys map to the same index.",
				"Common strategies include chaining and open addressing.",
				"Good hash functions reduce collision frequency.",
			],
			[
				"BFS is ideal for shortest paths in unweighted graphs.",
				"DFS is often used for exploration, backtracking, and cycle checks.",
				"Selection depends on the problem goal and memory constraints.",
			],
			[
				"Merge sort runs in O(n log n) time consistently.",
				"It splits the list recursively and merges sorted halves.",
				"Its predictable performance makes it reliable for large inputs.",
			],
		],
	),
	TopicTemplate(
		topic="Web and APIs",
		prompts=[
			"Explain REST API in beginner-friendly language.",
			"What is the role of HTTP status codes in API design?",
			"Why is authentication important in web APIs?",
			"Differentiate between PUT and PATCH methods.",
			"What is rate limiting and why do APIs use it?",
		],
		response_points=[
			[
				"A REST API lets applications communicate over HTTP using resource URLs.",
				"Clients send requests and servers return structured responses, often JSON.",
				"The design is simple, scalable, and language-agnostic.",
			],
			[
				"Status codes communicate request outcomes, like success or errors.",
				"They improve client-side handling and debugging.",
				"Consistent status code usage improves API reliability.",
			],
			[
				"Authentication verifies identity before granting access.",
				"It protects private data and prevents unauthorized usage.",
				"Common methods include tokens, API keys, and OAuth.",
			],
			[
				"PUT generally replaces an entire resource.",
				"PATCH updates only selected fields.",
				"PATCH is often more efficient for partial updates.",
			],
			[
				"Rate limiting controls how many requests a client can send.",
				"It prevents abuse and keeps services stable under load.",
				"It also supports fair usage across users.",
			],
		],
	),
	TopicTemplate(
		topic="Databases",
		prompts=[
			"What is normalization in relational databases?",
			"Explain primary key and foreign key with an example.",
			"Why do we use indexing in SQL databases?",
			"How does ACID improve transaction reliability?",
			"When would you choose NoSQL over SQL?",
		],
		response_points=[
			[
				"Normalization organizes tables to reduce redundancy and anomalies.",
				"It improves consistency through well-structured relationships.",
				"Over-normalization can be balanced with practical query performance needs.",
			],
			[
				"A primary key uniquely identifies rows in a table.",
				"A foreign key references another table's primary key.",
				"Together they enforce relational integrity.",
			],
			[
				"Indexes speed up lookups by creating efficient search structures.",
				"They reduce full table scans for frequent query patterns.",
				"Too many indexes can slow writes and increase storage usage.",
			],
			[
				"ACID stands for Atomicity, Consistency, Isolation, and Durability.",
				"These properties protect data correctness during failures or concurrency.",
				"They are essential for critical financial and enterprise systems.",
			],
			[
				"NoSQL is useful for flexible schemas and high horizontal scalability.",
				"SQL is stronger for complex joins and strict relational constraints.",
				"Choice depends on data shape, consistency needs, and query patterns.",
			],
		],
	),
	TopicTemplate(
		topic="Operating Systems",
		prompts=[
			"Explain process vs thread in OS.",
			"What is context switching and why is it expensive?",
			"How does virtual memory help modern systems?",
			"What is deadlock and how can it be prevented?",
			"Why is CPU scheduling important in an operating system?",
		],
		response_points=[
			[
				"A process has its own memory space and resources.",
				"Threads share memory within the same process.",
				"Threads are lighter but require careful synchronization.",
			],
			[
				"Context switching saves one task state and loads another.",
				"It introduces overhead because CPU caches and registers are affected.",
				"Frequent switches can reduce throughput.",
			],
			[
				"Virtual memory gives each process an isolated address space.",
				"It allows systems to use disk as an extension of RAM.",
				"This improves multitasking and memory safety.",
			],
			[
				"Deadlock occurs when processes wait on each other indefinitely.",
				"Prevention includes ordered resource acquisition and timeout strategies.",
				"Detection and recovery can also be used in controlled environments.",
			],
			[
				"Scheduling decides which process gets CPU time next.",
				"Good scheduling balances responsiveness, fairness, and throughput.",
				"Different algorithms suit different workloads.",
			],
		],
	),
]


def compose_response(points: Sequence[str], style_seed: int, concept_hint: str) -> str:
	"""Create varied but consistent responses from semantic points."""
	openers = [
		"In simple terms, ",
		"A practical way to see it is: ",
		"At a high level, ",
		"You can think of it this way: ",
		"From an interview perspective, ",
		"In day-to-day engineering work, ",
		"For a beginner, ",
		"In production systems, ",
	]
	closers = [
		"This is why the concept appears so often in real systems.",
		"That is the core idea used in interviews and production code.",
		"The main takeaway is to pick the approach based on context.",
		"Used correctly, this improves both reliability and performance.",
		"A short revision strategy is to connect the concept to one real use case.",
		"Once this intuition is clear, advanced details become easier to learn.",
	]
	examples = [
		f"For example, a small task around {concept_hint} can demonstrate the full flow clearly.",
		f"As a quick example, imagine handling {concept_hint} in a student project and validating the result.",
		f"One practical example is implementing {concept_hint} and measuring the effect before and after.",
		f"A concrete example would be reviewing {concept_hint} in a code walkthrough with teammates.",
	]
	followups = [
		"If needed, start with correctness first and optimize in a second step.",
		"Document assumptions so future changes stay safe and predictable.",
		"Always test edge cases to confirm the behavior matches your expectation.",
		"Use simple, readable implementations before adding extra abstraction.",
	]
	opener = openers[style_seed % len(openers)]
	closer = closers[style_seed % len(closers)]
	example = examples[(style_seed // 2) % len(examples)]
	followup = followups[(style_seed // 3) % len(followups)]
	return f"{opener}{' '.join(points)} {example} {followup} {closer}"


def pick_template_pair(rng: random.Random) -> Tuple[str, Sequence[str], str]:
	template = rng.choice(TEMPLATES)
	idx = rng.randrange(len(template.prompts))
	return template.prompts[idx], template.response_points[idx], template.topic


def diversify_instruction(base_instruction: str, topic: str, style_seed: int) -> str:
	prefixes = [
		"Explain",
		"Teach",
		"Describe",
		"Clarify",
		"Break down",
		"Summarize",
	]
	audience_tags = [
		"for a beginner",
		"for an interview prep learner",
		"for a final-year student",
		"for a junior developer",
		"for a quick revision",
		"with practical context",
	]
	constraints = [
		"in 5-7 lines.",
		"with one real-world example.",
		"with key points and pitfalls.",
		"with implementation intuition.",
		"in clear, simple language.",
		"without unnecessary theory.",
	]

	prefix = prefixes[style_seed % len(prefixes)]
	audience = audience_tags[(style_seed // 2) % len(audience_tags)]
	constraint = constraints[(style_seed // 3) % len(constraints)]

	trimmed = base_instruction.rstrip(".?")
	return f"{prefix} this {topic} concept: {trimmed}, {audience}, {constraint}"


def generate_raw_records(target_count: int, seed: int) -> List[DatasetItem]:
	rng = random.Random(seed)
	records: List[DatasetItem] = []

	for i in range(target_count):
		instruction, points, topic = pick_template_pair(rng)
		rich_instruction = diversify_instruction(instruction, topic=topic, style_seed=i + seed)
		response = compose_response(points, style_seed=i + seed, concept_hint=topic)
		records.append(
			{
				"instruction": rich_instruction,
				"input": "",
				"output": response,
			}
		)

	return records


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, content: List[DatasetItem]) -> None:
	ensure_parent_dir(path)
	with path.open("w", encoding="utf-8") as f:
		json.dump(content, f, ensure_ascii=False, indent=2)


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def run_phase1(
	target_count: int,
	seed: int,
	raw_output: Path,
	final_output: Path,
	min_output_words: int,
) -> Dict[str, int]:
	raw_records = generate_raw_records(target_count=target_count, seed=seed)
	cleaned_records = clean_dataset(raw_records, min_output_words=min_output_words)
	final_records = sort_dataset(to_instruction_schema(cleaned_records))

	save_json(raw_output, raw_records)
	save_json(final_output, final_records)

	return {
		"requested": target_count,
		"raw": len(raw_records),
		"cleaned": len(cleaned_records),
		"final": len(final_records),
	}


def build_arg_parser() -> argparse.ArgumentParser:
	root = project_root()

	parser = argparse.ArgumentParser(description="Phase 1 synthetic dataset generator")
	parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument(
		"--raw-output",
		type=Path,
		default=root / "data" / "raw" / "phase1_generated.json",
		help="Path to save raw generated records",
	)
	parser.add_argument(
		"--final-output",
		type=Path,
		default=root / "data" / "final_dataset.json",
		help="Path to save cleaned and formatted records",
	)
	parser.add_argument(
		"--min-output-words",
		type=int,
		default=12,
		help="Minimum output word count allowed by the quality filter",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	stats = run_phase1(
		target_count=args.count,
		seed=args.seed,
		raw_output=args.raw_output,
		final_output=args.final_output,
		min_output_words=args.min_output_words,
	)

	print("Phase 1 dataset generation complete")
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()

