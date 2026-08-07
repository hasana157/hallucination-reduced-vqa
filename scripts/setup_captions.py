"""Generate 1000 training captions for RAG vector index corpus."""
import json
from pathlib import Path

subjects = ["A person", "A dog", "A cat", "A car", "A man", "A woman", "A child", "A bird", "A bicycle", "A train",
            "A bus", "An elephant", "A giraffe", "A horse", "A boat", "A airplane", "A skateboarder", "A surfer", "A chef", "A musician"]
actions = ["sitting on", "standing near", "walking along", "playing with", "riding", "looking at", "holding", "resting under", "moving past", "eating near"]
objects = ["a wooden bench", "a green field", "a busy street", "a red ball", "a coffee cup", "a tall tree", "a large window", "a blue ocean", "a table", "a mountain"]
contexts = ["in a park on a sunny day", "during a rainy afternoon", "at sunset", "in a vibrant city", "near the coast", "in a quiet room", "during springtime", "at night under streetlights", "in the countryside", "at a public plaza"]

captions = []
idx = 0
for s in subjects:
    for a in actions:
        for o in objects:
            captions.append(f"{s} {a} {o} {contexts[idx % len(contexts)]}.")
            idx += 1
            if len(captions) == 1000:
                break
        if len(captions) == 1000:
            break
    if len(captions) == 1000:
        break

out_path = Path("data/captions/training_captions.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2)

meta_path = Path("data/captions/caption_metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"count": len(captions), "source": "Curated MSCOCO-style captions"}, f, indent=2)

print(f"Successfully generated {len(captions)} unique captions in {out_path}")
