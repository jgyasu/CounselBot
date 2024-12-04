import requests
from bs4 import BeautifulSoup
import os
import shutil
from pathlib import Path

base_path = Path(__file__).resolve().parent
documents_path = base_path / 'documents'

def scrape_rulebooks(url, save_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    
    for link in links:
        if link['href'].endswith('.pdf'):
            pdf_url = link['href']
            if not pdf_url.startswith('http'):
                pdf_url = url + pdf_url
            response = requests.get(pdf_url)
            filename = os.path.join(save_path, pdf_url.split('/')[-1])
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            
# Example Usage
scrape_rulebooks("https://josaa.nic.in/", documents_path)
# scrape_rulebooks("https://josaa.nic.in/restrictions-at-institutes-academic-programs-level/", documents_path)
scrape_rulebooks("https://josaa.nic.in/information-bulletin/", documents_path)
# scrape_rulebooks("https://csab.nic.in/", documents_path)
# scrape_rulebooks("https://mcc.nic.in/ug-medical-counselling/", documents_path)