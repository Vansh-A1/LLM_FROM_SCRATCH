import json
from pathlib import Path

from datasets import load_dataset


NUM_DOCUMENTS = 100

OUTPUT_DIR = Path(
    "/data/vansh/LLM_FROM_SCRATCH/LLM_CODE/data/raw/fineweb"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fineweb_edu_pilot_100.jsonl"


print("Connecting to FineWeb-Edu...")


dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)


print(f"Downloading first {NUM_DOCUMENTS} documents...")


count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for document in dataset:

        # Save the complete record exactly as provided
        # by the Hugging Face dataset interface.
        f.write(
            json.dumps(
                document,
                ensure_ascii=False
            )
            + "\n"
        )

        count += 1

        if count % 10 == 0:
            print(f"Saved {count}/{NUM_DOCUMENTS}")

        if count >= NUM_DOCUMENTS:
            break


print()
print("FineWeb-Edu pilot download complete.")
print(f"Documents saved: {count}")
print(f"Output: {OUTPUT_FILE}")