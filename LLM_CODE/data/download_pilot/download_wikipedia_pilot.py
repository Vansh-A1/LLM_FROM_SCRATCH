import json
import time
from pathlib import Path

import requests


API_URL = "https://en.wikipedia.org/w/api.php"

OUTPUT_DIR = Path("data/raw/wikipedia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "LLMFromScratchPilot/0.1 (educational research project)"
}

NUM_ARTICLES = 100
BATCH_SIZE = 20

random_params = {
    "action": "query",
    "format": "json",
    "list": "random",
    "rnnamespace": 0,     
    "rnfilterredir": "nonredirects",
    "rnlimit": NUM_ARTICLES,
}

response = requests.get(
    API_URL,
    params=random_params,
    headers=HEADERS,
    timeout=60,
)

response.raise_for_status()

random_data = response.json()


with open(OUTPUT_DIR / "random_selection.json", "w", encoding="utf-8") as f:
    json.dump(random_data, f, ensure_ascii=False, indent=2)


titles = [
    page["title"]
    for page in random_data["query"]["random"]
]

print(f"Selected {len(titles)} Wikipedia articles.")

for batch_number, start in enumerate(
    range(0, len(titles), BATCH_SIZE)
):

    batch_titles = titles[start:start + BATCH_SIZE]

    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,

        "prop": "revisions",

        "titles": "|".join(batch_titles),

    
        "rvprop": "ids|timestamp|content",

        "rvslots": "main",
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    output_file = (
        OUTPUT_DIR
        / f"articles_{batch_number:03d}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved batch {batch_number}: "
        f"{len(batch_titles)} articles"
    )

    time.sleep(1)


print("Wikipedia pilot download complete.")