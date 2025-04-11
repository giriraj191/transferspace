import re
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.parse import urlparse
from src.modules.scraper import scrape_website
import warnings
warnings.simplefilter("ignore")


def generate_directory_name(url):
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', url)
    return clean_name[:70].lower() if clean_name else "website"


def extract_links(soup, base_url):
    base_domain = urlparse(base_url).netloc  # Extract domain
    links = set()
    
    for body in soup.find_all("body"):
        for tag in body.find_all(["p", "li"]):
            for a_tag in tag.find_all("a", href=True):
                link = urljoin(base_url, a_tag["href"])
                if "#" not in link and "?" not in link and urlparse(link).netloc == base_domain:
                    links.add(link)
    
    return list(links)


def handle_link(start_url, depth, num_links):
    visited_links = set()
    queue = [(start_url, 0)]

    website_name = generate_directory_name(start_url)
    BASE_DIR = os.path.join("artifacts", "raw")

    while queue:
        url, current_depth = queue.pop(0)
        
        if url in visited_links or current_depth > depth:
            continue
        visited_links.add(url)
        
        print(f"\nscraping: {url} at depth {current_depth}")
        scrape_website(url, current_depth, BASE_DIR, website_name)
        
        if current_depth < depth:
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                new_links = extract_links(soup, start_url)
                print(f"total extracted sub-links for current link: {len(new_links)}")
                new_links = new_links[:num_links]
                queue.extend([(link, current_depth + 1) for link in new_links if link not in visited_links])


def handle_links(urls, depth, num_links):
    directory_names = []
    for url in urls:
        directory_names.append(generate_directory_name(url))
    for url in urls:
        print(f"\ninitiating process | root link: {url}")
        handle_link(url, depth, num_links)
    return directory_names