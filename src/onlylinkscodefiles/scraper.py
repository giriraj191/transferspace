import os
import re
import json
import requests
from tqdm import tqdm
from urllib.parse import urljoin
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from unstructured.partition.html import partition_html
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import warnings
warnings.simplefilter("ignore")


def valid_url_name(url):
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', url)
    return clean_name[:70].lower() if clean_name else "website"


def detect_language(code_snippet):
    code_snippet = code_snippet.lower().strip()

    if code_snippet.startswith('"') and code_snippet.endswith('"') and '\\n' in code_snippet:
        return "markdown/blockquote"

    lang_keywords = {
        "python": [
            "def ", "import ", "print(", "lambda ", "class ", "async ", "await ",
            "elif ", "raise ", "except:", "try:", "with ", "@decorator",
            "llm = ", "workflow = ", ".invoke(", ": State", "# Set our",
            "model_path=", "callback_manager=", "verbose=", "n_gpu_layers=",
            "n_batch=", "n_ctx=", "from", "langchain", "langchain_core", "langchain_community",
            "prompt", "AIAgent", "HumanMessage", "SystemMessage", "messages"
        ],
        
        "javascript": [
            "function ", "console.log(", "let ", "const ", "var ", "export ",
            "import ", "=>", "addEventListener", "querySelector", "async function",
            "Promise", "fetch("
        ],
        
        "typescript": [
            "interface ", "type ", ": string", ": number", ": boolean",
            "implements ", "extends ", "namespace ", ": void", ": any"
        ],
        
        "html": [
            "<!doctype html>", "<html>", "<body>", "<div", "<p>", "<script",
            "<style", "<span", "<head>", "<meta", "<link", "<img", "<form"
        ],
        
        "css": [
            "@media", "background-color:", "margin:", "padding:", "display: flex",
            "border-radius:", "@keyframes", "!important", "font-family:"
        ],
        
        "java": [
            "public class", "system.out.println", "void main", "extends ",
            "implements ", "import java.", "private ", "protected "
        ],
        
        "c++": [
            "#include", "cout <<", "cin >>", "int main", "std::", "namespace ",
            "vector<", "template<", "::", "public:", "private:"
        ],
        
        "sql": [
            "select ", "from ", "where ", "join ", "group by ",
            "having ", "order by ", "insert into", "update ", "delete from "
        ],
        
        "shell": [
            "#!/bin/", "echo ", "sudo ", "chmod ", "export ",
            "source ", "alias ", "if [ ", "for i in", "pip", "install"
        ]
    }

    # counting scores to return the best match
    language_scores = {}
    
    for lang, keywords in lang_keywords.items():
        score = sum(1 for keyword in keywords if keyword.lower() in code_snippet)
        if score > 0:
            language_scores[lang] = score
    
    if not language_scores:
        return "unknown"
    
    detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]
    
    # handling json and javascript
    if detected_lang == "javascript" and "function" not in code_snippet.lower():
        if code_snippet.strip().startswith(("{", "[")):
            return "json"
    
    return detected_lang


def save_images(images, IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    image_links = []
    for i, img_data in tqdm(enumerate(images)):
        img_url = img_data["url"]
        img_url_name = valid_url_name(img_url)
        try:
            img_response = requests.get(img_url, timeout=5)
            img_response.raise_for_status()
            img_content = Image.open(BytesIO(img_response.content))
            img_ext = os.path.splitext(img_url.split("?")[0])[-1].lower()
            img_ext = img_ext if img_ext in [".jpg", ".jpeg", ".png"] else ".png"
            img_filename = os.path.join(IMAGE_DIR, f"{img_url_name}{img_ext}")
            if img_ext in [".jpg", ".jpeg", ".png"] and img_content.mode == "RGBA":
                img_content = img_content.convert("RGB")
            img_content.save(img_filename)
            image_links.append(img_filename)
        except UnidentifiedImageError:
            pass
        except Exception as e:
            pass
    return image_links


def scrape_website(url, level, BASE_DIR, directory_name):
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return {"error": f"Failed to fetch {url}"}
    
    # creating respective directories for the website
    website_name = directory_name
    url_name = valid_url_name(url)
    base_dir = os.path.join(BASE_DIR, website_name)
    json_dir = os.path.join(base_dir, "json")
    image_dir = os.path.join(base_dir, "images")
    table_dir = os.path.join(base_dir, "tables")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    soup = BeautifulSoup(response.content, "html.parser")
    print(f"extracting text")
    text_data = " ".join([data.get_text(strip=True) for data in soup.find_all(["p", "h1", "h2", "h3", "li"])])
    elements = partition_html(text=response.text)
    text_data += "\n".join([str(el) for el in elements if el.category == "NarrativeText"])
    
    # handling and saving tables
    print(f"saving tables")
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        rows = [[cell.get_text(strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        if rows:
            df = pd.DataFrame(rows)
            table_path = os.path.join(table_dir, f"{url_name}_table_{i+1}.csv")
            df.to_csv(table_path, index=False)
            tables.append(table_path)
    
    # saving images
    print(f"saving images")
    images = [{"url": urljoin(url, img.get("src"))} for img in soup.find_all("img") if img.get("src")]
    image_links = save_images(images, image_dir)
    
    print(f"extracting code blocks")
    # working with code blocks
    code_blocks = []
    for code in soup.find_all(["code", "pre"]):
        code_text = code.get_text(strip=True)
        if len(code_text) > 10:
            language = detect_language(code_text)
            code_blocks.append({"language": language, "content": code_text})

    data = {
        "source": url,
        "level": level,
        "wname": website_name,
        "text": text_data,
        "tables": table_dir if tables else None,
        "images": image_links if image_links else None,
        "code_blocks": code_blocks if code_blocks else None
    }
    
    # saving json file for the current URL
    print(f"saving json")
    json_filename = os.path.join(json_dir, f"scraped_{url_name}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)