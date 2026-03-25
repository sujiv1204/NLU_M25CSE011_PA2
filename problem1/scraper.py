import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pypdf

def fetch_webpage(url):
    """Fetches the HTML content from a given URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def extract_text(html):
    """Strips out HTML tags, scripts, and non-ASCII character noise."""
    if not html:
        return ""
    
    # Utilizing BeautifulSoup to safely drop structural tags that don't hold conversational semantic value
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
        
    text = soup.get_text(separator=' ')
    
    # Regex sweeps to collapse enormous white-space gaps often left behind by removed DOM elements
    text = re.sub(r'\s+', ' ', text)
    # Aggressively filter out non-ASCII characters since our downstream pipeline targets clean English tokens
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def download_pdf(url, save_path):
    """Downloads a PDF and immediately extracts its textual content."""
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=60)
        is_pdf = 'pdf' in response.headers.get('content-type', '').lower() or url.endswith('.pdf')
        if response.status_code == 200 and is_pdf:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            extracted_length = 0
            text = ""
            try:
                with open(save_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                extracted_length = len(text)
            except Exception as e:
                pass 
                
            return text if extracted_length > 0 else None
    except Exception as e:
        pass
    return None

def run_scraper():
    all_texts = []
    visited_urls = set()

    standard_queue = ["https://iitj.ac.in/"]
    
    # Exactly original target seed links provided previously for guaranteed accuracy
    priority_queue = [
        "https://www.iitj.ac.in/office-of-academics/en/academic-regulations", 
        "https://www.iitj.ac.in/main/en/faculty-members",
        "https://www.iitj.ac.in/m/Index/main-institute?lg=en",
        "https://www.iitj.ac.in/office-of-academics/en/academics",
        "https://www.iitj.ac.in/main/en/institute-reports",
        "https://www.iitj.ac.in/m/Index/main-outreach?lg=en",
        "https://www.iitj.ac.in/m/Index/main-schools?lg=en",
        
    ]
    
    departments = ["computer-science-engineering", "mechanical-engineering", "electrical-engineering", "mathematics", "physics", "chemistry", "bioscience-bioengineering"]
    for dept in departments:
        priority_queue.append(f"https://www.iitj.ac.in/{dept}/en/faculty")

    pdf_directory = "data/pdfs"
    os.makedirs(pdf_directory, exist_ok=True)

    print("Starting Web Scraper")
    
    while (priority_queue or standard_queue) and len(visited_urls) < 500:
        if priority_queue:
            url = priority_queue.pop(0)
        else:
            url = standard_queue.pop(0)
            
        if url in visited_urls:
            continue
            
        visited_urls.add(url)
        print(f"fetching {len(visited_urls)}/500: {url[:70]}...", flush=True)
        
        if url.lower().endswith('.pdf'):
            safe_filename = url.split('/')[-1].split('?')[0]
            local_path = os.path.join(pdf_directory, f"{len(visited_urls)}_{safe_filename}")
            
            pdf_text = download_pdf(url, local_path)
            if pdf_text and len(pdf_text) > 100:
                all_texts.append(pdf_text)
                print(f"  got valid pdf with {len(pdf_text)} chars", flush=True)
            continue

        html_content = fetch_webpage(url)
        if not html_content:
            continue
            
        page_text = extract_text(html_content)
        if page_text and len(page_text) > 150:
            all_texts.append(page_text)
            print(f"  got {len(page_text)} chars from page", flush=True)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href']).split('#')[0]
                
                allowed_domains = ['iitj.ac.in', 'sites.google.com', 'github.io']
                domain_allowed = any(domain in urlparse(next_url).netloc for domain in allowed_domains)
                
                disallowed_extensions = ('.png', '.jpg', '.jpeg', '.zip', '.mp4', '.css', '.js')
                if domain_allowed and next_url not in visited_urls and not next_url.endswith(disallowed_extensions):
                    blocklist = ['lecture', 'note', 'slide', 'assignment', 'homework', 'exam', 'quiz', 'solution']
                    if any(blocked in next_url.lower() for blocked in blocklist):
                        continue

                    hot_keywords = ['regulation', 'syllabus', 'course', 'circular', 'newsletter', 'research', 'academic', 'program', 'faculty', 'department', '.pdf', 'ug', 'pg', 'undergraduate', 'postgraduate', 'btech', 'mtech', 'phd', 'curriculum']
                    if any(target in next_url.lower() for target in hot_keywords):
                        priority_queue.append(next_url)
                    else:
                        standard_queue.append(next_url)
        else:
            print("  no content", flush=True)
            
        time.sleep(0.05)

    print(f"Collected total raw documents: {len(all_texts)}")
    
    with open('data/raw_corpus.txt', 'w', encoding='utf-8') as f:
        for document in all_texts:
            f.write(document + '\n\n')
            
    print("Saved raw corpus cleanly")

if __name__ == "__main__":
    run_scraper()
