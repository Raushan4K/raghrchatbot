import os
import pickle

def chunk_text(text, chunk_size=400, overlap=60):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    input_path = os.path.join("..", "data", "hr_policy_clean.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    # Save chunks for embedding step
    with open(os.path.join("..", "data", "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
