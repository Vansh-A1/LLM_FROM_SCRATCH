import json
from pathlib import Path

from datasets import load_dataset


NUM_DOCUMENTS = 100

OUTPUT_DIR = Path(
    "/data/vansh/LLM_FROM_SCRATCH/LLM_CODE/data/raw/code"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "stackv2_edu_pilot_100.jsonl"


print("Connecting to Stack V2 Edu...")


dataset = load_dataset(
    "common-pile/stackv2_edu_filtered",
    split="train",
    streaming=True,
)


print(f"Downloading first {NUM_DOCUMENTS} code documents...")


count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for document in dataset:

        # Keep the complete raw record.
        f.write(
            json.dumps(
                document,
                ensure_ascii=False,
                default=str
            )
            + "\n"
        )

        count += 1

        if count % 10 == 0:
            print(f"Saved {count}/{NUM_DOCUMENTS}")

        if count >= NUM_DOCUMENTS:
            break


print()
print("Code pilot download complete.")
print(f"Documents saved: {count}")
print(f"Output: {OUTPUT_FILE}")