import os
import json
import glob
import yaml
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def chunk_markdown(content, filename):
    parts = re.split(r'\n## ', '\n' + content)
    
    chunks = []
    h1 = ""
    for i, part in enumerate(parts):
        if not part.strip(): continue
        
        if i == 0:
            match = re.search(r'^#\s+(.+)$', part, re.MULTILINE)
            if match:
                h1 = match.group(1).strip()
            chunks.append({"heading": h1 or filename, "content": part.strip()})
        else:
            lines = part.split('\n', 1)
            h2 = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            heading = f"{h1} - {h2}" if h1 else h2
            full_content = f"## {h2}\n{body}"
            chunks.append({"heading": heading, "content": full_content.strip()})
            
    return chunks

def index_knowledge_base(kb_dir=None, output_file=None):
    kb_dir = kb_dir or str(PROJECT_ROOT / "knowledge-base")
    output_file = output_file or str(PROJECT_ROOT / "data" / "index.json")
    documents = []
    
    for filepath in glob.glob(os.path.join(kb_dir, "*.md")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        metadata = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1]) or {}
                except Exception as e:
                    print(f"Error parsing YAML in {filename}: {e}")
                content = parts[2].strip()
                
        chunks = chunk_markdown(content, filename)
        
        for idx, chunk in enumerate(chunks):
            # Text to embed: include context so the embedding captures what this is about
            text_to_embed = f"Document: {metadata.get('title', filename)}\nHeading: {chunk['heading']}\n\n{chunk['content']}"
            
            try:
                response = client.models.embed_content(
                    model='gemini-embedding-2',
                    contents=text_to_embed
                )
                embedding = response.embeddings[0].values
                
                documents.append({
                    "id": f"{filename}-{idx}",
                    "filename": filename,
                    "metadata": metadata,
                    "heading": chunk['heading'],
                    "content": chunk['content'],
                    "embedding": embedding
                })
                print(f"Embedded {filename} - {chunk['heading']}")
            except Exception as e:
                print(f"Failed to embed {filename} - {chunk['heading']}: {e}")
                
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, default=str)
        
    print(f"Indexed {len(documents)} chunks to {output_file}")

if __name__ == "__main__":
    index_knowledge_base()
