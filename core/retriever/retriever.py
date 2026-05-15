import requests
import re
import logging
import time
import traceback
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger("Retriever")

# Known low-quality / non-authoritative domains and patterns
LOW_QUALITY_DOMAINS = {
    # Social media & forums
    "reddit.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "quora.com", "answers.com", "yahoo.com/answers",
    "medium.com", "stackexchange.com", "stackoverflow.com",
    # Shopping & commercial
    "amazon.com", "ebay.com", "aliexpress.com", "walmart.com", "etsy.com",
    "alibaba.com", "shopify.com", "bestbuy.com",
    # Video & image
    "youtube.com", "vimeo.com", "dailymotion.com", "pinterest.com",
    "flickr.com", "imgur.com",
    # News (not peer-reviewed)
    "bbc.com", "cnn.com", "foxnews.com", "nytimes.com", "theguardian.com",
    "washingtonpost.com", "usatoday.com", "reuters.com", "apnews.com",
    "bloomberg.com", "wsj.com", "huffpost.com", "buzzfeed.com",
    # Low-credibility health sites
    "webmd.com", "healthline.com", "verywellhealth.com", "medicinenet.com",
    "draxe.com", "mercola.com", "naturalnews.com", "drweil.com",
    # General reference (not primary sources)
    "wikipedia.org", "britannica.com", "wikihow.com", "about.com",
    # Commercial biotech/lab suppliers
    "thermofisher.com", "sigmaaldrich.com", "acrobiosystems.com",
    "abcam.com", "biolegend.com", "fishersci.com", "neb.com",
    # Ad/tracking domains
    "bing.com/aclick", "doubleclick.net", "googlesyndication.com",
}

# File extensions that indicate non-HTML content
NON_TEXT_EXTENSIONS = (
    ".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".gif",
    ".zip", ".docx", ".pptx", ".mp4", ".mp3", ".avi", ".mov",
    ".csv", ".ppt", ".doc", ".epub", ".mobi", ".svg", ".webp",
)


def is_low_quality_url(url):
    """Check if a URL belongs to a known low-quality domain."""
    url_lower = url.lower()
    for domain in LOW_QUALITY_DOMAINS:
        if domain in url_lower:
            return True
    return False


def has_non_text_extension(url):
    """Check if URL points to a non-HTML file."""
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in NON_TEXT_EXTENSIONS)


def score_document_relevance(paragraphs, keywords):
    """Score how relevant a document is to the target keywords (0.0 - 1.0).
    Returns 0.0 for documents that contain none of the keywords."""
    if not paragraphs or not keywords:
        return 0.0
    full_text = " ".join(paragraphs).lower()
    hits = sum(1 for k in keywords if k.lower() in full_text)
    return hits / len(keywords)


def process_text(text):
    """Clean text by removing excessive whitespaces and correcting common encoding artifacts."""
    text = re.sub(r'[\n\t]+', ' ', text)
    text = text.replace("â€™", "'").replace(" ", " ")
    return text.strip()


def combine_adjacent_paragraphs(soup, tag, keywords=None, max_total_chars=6000):
    """
    Extract and prune text from HTML tags. Priority:
    1. Intro paragraphs
    2. Paragraphs containing domain keywords
    3. Concluding paragraph
    Also filters out navigation/menu noise.
    """
    paragraphs = soup.find_all(tag)
    combined_paragraphs = []
    current_paragraph = ""

    for i in range(len(paragraphs)):
        if i == 0:
            current_paragraph = paragraphs[i].get_text()
        else:
            if paragraphs[i].previous_sibling == paragraphs[i - 1]:
                current_paragraph += " " + paragraphs[i].get_text()
            else:
                combined_paragraphs.append(process_text(current_paragraph))
                current_paragraph = paragraphs[i].get_text()

    if current_paragraph:
        combined_paragraphs.append(process_text(current_paragraph))

    min_word_num = 3
    # Filter out very short fragments (navigation, menus, etc.)
    long_paragraphs = [s for s in combined_paragraphs if len(s.split()) >= min_word_num]

    # Also filter out paragraphs that look like navigation/banners (high ratio of short words or special chars)
    nav_noise_pattern = re.compile(r'^[\s\W\d]+$')
    content_paragraphs = [s for s in long_paragraphs if not nav_noise_pattern.match(s) and len(s) >= 20]

    if not content_paragraphs:
        return []

    seen = set()
    cleaned_paragraphs = []
    for s in content_paragraphs:
        normalized = s.lower()[:100]  # Dedup by first 100 chars (lowercase)
        if normalized not in seen:
            seen.add(normalized)
            cleaned_paragraphs.append(s)

    if not cleaned_paragraphs:
        return []

    final_paragraphs = []
    current_total_len = 0

    # Priority A: Keep introductory content
    for p in cleaned_paragraphs[:2]:
        if current_total_len + len(p) < max_total_chars:
            final_paragraphs.append(p)
            current_total_len += len(p)

    # Priority B: Keep keyword-relevant paragraphs
    if keywords:
        for p in cleaned_paragraphs[2:-1]:
            if any(k.lower() in p.lower() for k in keywords):
                if current_total_len + len(p) < max_total_chars:
                    final_paragraphs.append(p)
                    current_total_len += len(p)
            if current_total_len >= max_total_chars:
                break

    # Priority C: Keep the concluding segment
    last_p = cleaned_paragraphs[-1]
    if last_p not in final_paragraphs and current_total_len + len(last_p) < max_total_chars:
        final_paragraphs.append(last_p)

    return final_paragraphs


def deep_retrieve_by_authorities(query, domains, extracted_docs, keywords, proxy_url,
                                  max_new_docs=10, timeout=5, pages=1, min_relevance=0.1):
    """Targeted search via DuckDuckGo restricted to authoritative domains."""
    logger.info(f"Targeted search: {query}")
    all_retrieved_results = []

    with DDGS(proxy=proxy_url) as ddgs:
        for domain in domains[:4]:
            try:
                results = ddgs.text(f'{query} site:{domain}', region='wt-wt', max_results=pages * 3)
                if results:
                    for r in results:
                        all_retrieved_results.append({'link': r['href'], 'title': r['title']})
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Search failed for {domain}: {e}")
                logger.debug(traceback.format_exc())

    new_docs_count = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for item in all_retrieved_results:
        if new_docs_count >= max_new_docs:
            break
        url = item['link']

        # URL quality pre-checks
        if not url or not url.startswith("http"):
            continue
        if has_non_text_extension(url):
            continue
        if is_low_quality_url(url):
            logger.debug(f"Skipping low-quality URL: {url}")
            continue
        if url in extracted_docs:
            continue

        try:
            logger.debug(f"Scraping: {url}")
            response = requests.get(url, headers=headers, timeout=timeout,
                                    proxies={"http": proxy_url, "https": proxy_url})
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')
            combined_paragraphs = combine_adjacent_paragraphs(soup, 'p', keywords=keywords)

            if not combined_paragraphs:
                continue

            # Content relevance check: skip docs with zero keyword matches
            relevance = score_document_relevance(combined_paragraphs, keywords)
            if relevance < min_relevance:
                logger.debug(f"Skipping irrelevant doc (relevance={relevance:.2f}): {url}")
                continue

            extracted_docs[url] = combined_paragraphs
            new_docs_count += 1
            logger.debug(f"Accepted doc (relevance={relevance:.2f}): {url}")

        except requests.Timeout:
            logger.warning(f"Timeout scraping {url} (timeout={timeout}s)")
        except requests.ConnectionError:
            logger.warning(f"Connection failed for {url}")
        except requests.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            logger.debug(traceback.format_exc())

    return extracted_docs
