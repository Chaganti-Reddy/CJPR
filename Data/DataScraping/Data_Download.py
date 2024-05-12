import json
import os
import time
import urllib.request
from urllib.error import HTTPError

import progressbar
from bs4 import BeautifulSoup

with open('links_Supreme_Court.json', 'r') as file:
    links_json = json.load(file)
category = 'Supreme Court of India'
json_file = 'links_Supreme_Court.json'
category_links = links_json[category]
ndocs = 0

with open('links_Supreme_Court.json', 'r') as json_file1:
    data = json.load(json_file1)

urls = data["Supreme Court of India"]

for url in urls:
    year = url[-4:]
    print("Processing Year:", year)
    if not os.path.exists('Yearwise_data'):
        os.mkdir('Yearwise_data')
    year_directory = os.path.join('Yearwise_data', year)
    if not os.path.exists(year_directory):
        os.mkdir(year_directory)
    print("Year processed.\n")

print("Started file {0} with docs = {1}\n".format(json_file, len(links_json[category])))

k = 1947
tmp = 0
while len(category_links) > 0:
    time.sleep(2)
    links_done_in_this_loop = []
    for i in (progressbar.progressbar(range(len(category_links)))):
        BASE_URL = category_links[i]
        year = int(category_links[i][-4:])
        # links_done_in_this_loop.append(BASE_URL)
        if year >= 1947:
            if year >= k + 1:
                tmp += ndocs
                ndocs = 0
                k = year
            path = f'Yearwise_data/{year}/{year}_{ndocs}'
            if os.path.exists(path):
                print(f'{year}_{ndocs} already exists, skipping...')
                ndocs = ndocs + 1
                links_done_in_this_loop.append(BASE_URL)
            else:
                while True:  # Retry loop
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                        req = urllib.request.Request(BASE_URL, headers=headers)
                        html = urllib.request.urlopen(req).read()
                    except urllib.error.HTTPError as e:
                        if e.code == 429:  # Too Many Requests
                            print("Too many requests. Retrying...\n")
                            time.sleep(5)  # Wait for 5 seconds
                        else:
                            print(f"An HTTP error occurred: {e}")
                            print(f"Status code: {e.code}")
                            print("Occurred at doc", ndocs)
                            time.sleep(3)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        time.sleep(3)
                    else:
                        break  # Break the retry loop if successful
                soup = BeautifulSoup(html, "html.parser")
                data_html = soup.find("div", attrs={"class": "judgments"})
                text = data_html.get_text()
                with open(path, "wb+") as f:
                    try:
                        f.write(text.encode('utf-8'))
                    except UnicodeEncodeError as e:
                        print(f"Encoding error: {e}")
                print(f'\n{year}_{ndocs} is saved...')
                ndocs = ndocs + 1
                links_done_in_this_loop.append(BASE_URL)
                if ndocs % 100 == 0:
                    time.sleep(5)

    for link in links_done_in_this_loop:
        if link in category_links:
            category_links.remove(link)
        else:
            print("Link not found in the list.")

    print("Docs that were downloaded: {0}\n\n\n".format(tmp + ndocs))
