import os
import math
import urllib.parse
import tqdm
import requests
import pandas as pd


def download_dataset(query, apiKey, images_dir, maxImages=50, force=False):
    """Builds a dataset and returns a dataframe index."""
    assert maxImages <= 500, "Only 500 images can be downloaded."
    os.makedirs(images_dir, exist_ok=True)
    pageSize = min(200, maxImages)
    sanitized = urllib.parse.quote(query, safe="")
    base = "https://pixabay.com/api/?key={apiKey}&q={query}&per_page={pageSize}&page={page}"
    first = requests.get(
        base.format(apiKey=apiKey, query=sanitized, pageSize=pageSize, page=1),
        timeout=10,
    ).json()
    items = first["hits"]
    pages = math.ceil(min(first["totalHits"], maxImages) / pageSize)
    for page in range(2, pages + 1):
        items.extend(
            requests.get(
                base.format(
                    apiKey=apiKey, query=sanitized, pageSize=pageSize, page=page
                ),
                timeout=10,
            ).json()["hits"]
        )
    for item in tqdm.tqdm(items):
        url = item["webformatURL"]
        filename = url.split("/")[-1]
        item["filepath"] = os.path.join(images_dir, filename)
        if not os.path.isfile(item["filepath"]) or force:
            with open(item["filepath"], "wb") as f:
                f.write(requests.get(url, allow_redirects=True, timeout=10).content)
    return pd.DataFrame(items)
