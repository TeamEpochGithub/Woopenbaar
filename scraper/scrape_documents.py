import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from scraper.website_specific import (
    website_to_pdf,
    extract_pdf_from_tweedekamer,
    handle_openkamer_site,
    handle_officielebekendmakingen,
)
from standard_data_format.src.metadata import MetadataManager


def is_excluded_domain(url):
    """
    Check if the URL belongs to a domain that should be excluded (social media, search engines, etc.)

    Args:
        url: URL to check

    Returns:
        bool: True if the domain should be excluded, False otherwise
    """
    excluded_domains = [
        # Social media
        "facebook.com",
        "fb.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "pinterest.com",
        "reddit.com",
        "snapchat.com",
        "whatsapp.com",
        "telegram.org",
        "t.me",
        # Search engines
        "google.com",
        "google.nl",
        "bing.com",
        "yahoo.com",
        "duckduckgo.com",
        "baidu.com",
        "yandex.com",
        # Other platforms not relevant for document scraping
        "spotify.com",
        "apple.com",
        "amazon.com",
        "netflix.com",
        "microsoft.com",
        "office.com",
        "live.com",
        "outlook.com"
        # Dutch tv sites\
        "npostart.nl",
        "geenstijl.nl",
        "nporadio1.nl",
    ]

    domain = extract_domain(url)

    # Check if any of the excluded domains is in the URL's domain
    for excluded in excluded_domains:
        if excluded in domain:
            return True

    return False


def scrape_public_links(
    df,
    output_dir="data/scraped_documents_v2",
    failed_links_file="data/failed_links.csv",
):
    """
    Scrape documents from public links in the dataframe, prioritizing PDF downloads

    Args:
        df: DataFrame containing document links
        output_dir: Directory to save downloaded documents
        failed_links_file: Path to save CSV file with failed links
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0
    excluded = 0

    # Track failures by domain
    failures_by_domain = {}

    # Track failed links for later review
    failed_links_data = []

    # Define headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    # Custom headers for specific domains
    domain_specific_headers = {
        "tweedemonitor.nl": {
            "Referer": "https://www.tweedemonitor.nl/",
        },
        "sec.gov": {
            "Referer": "https://www.sec.gov/",
        },
        "officielebekendmakingen.nl": {
            "Referer": "https://zoek.officielebekendmakingen.nl/",
        },
    }

    # Try different possible column names for the document link
    link_columns = ["Publieke Link", "Publieke link"]
    link_column = None

    for col in link_columns:
        if col in df.columns:
            link_column = col
            print(f"Using '{link_column}' column for document links")
            break

    if not link_column:
        print(f"Error: No link column found. Available columns: {df.columns.tolist()}")
        return 0, 0, 0, 0, {}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            document_links_raw = row[link_column]
            document_id = row.get("ID", f"doc_{index}")

            # Skip if document already exists in output directory
            existing_files = glob.glob(os.path.join(output_dir, f"{document_id}.*"))
            if existing_files:
                print(f"Skipping document {document_id} - already downloaded")
                skipped += 1
                continue

            if pd.isna(document_links_raw) or not document_links_raw:
                print(f"Skipping document at index {index}: No link available")
                continue

            # Handle multiple links in a single field
            # Split by common separators: |, ;, newlines
            document_links = [
                link.strip() for link in re.split(r"[|;\n]+", document_links_raw)
            ]

            # Try each link until one succeeds
            success = False
            link_errors = []

            for document_link in document_links:
                try:
                    # Clean the URL - strip whitespace and fix common issues
                    document_link = document_link.strip()
                    # Remove trailing %20 (URL-encoded space) that can cause 404 errors
                    if document_link.endswith("%20"):
                        document_link = document_link[:-3]

                    # Check if the domain should be excluded
                    if is_excluded_domain(document_link):
                        print(f"Excluding link: {document_link} (excluded domain)")
                        link_errors.append(
                            f"Excluded domain: {extract_domain(document_link)}"
                        )
                        excluded += 1
                        continue

                    print(
                        f"Processing document {document_id}, trying link: {document_link}"
                    )

                    # Get domain for site-specific handling
                    domain = extract_domain(document_link)

                    # Prepare request headers - combine default with domain-specific
                    request_headers = headers.copy()
                    if domain in domain_specific_headers:
                        request_headers.update(domain_specific_headers[domain])

                    # Special handling for openkamer.org and similar sites
                    if domain == "openkamer.org" or "kamervraag" in document_link:
                        print(
                            "Detected openkamer.org or kamervraag link, looking for source document links"
                        )
                        pdf_link = handle_openkamer_site(document_link, request_headers)
                        if pdf_link:
                            print(f"Found PDF link from openkamer: {pdf_link}")
                            # Update document_link to the PDF link
                            document_link = pdf_link
                            # Update domain and headers for the new link
                            domain = extract_domain(document_link)
                            request_headers = headers.copy()
                            if domain in domain_specific_headers:
                                request_headers.update(domain_specific_headers[domain])

                    # Check if the URL is directly a PDF
                    if document_link.lower().endswith(".pdf"):
                        print(f"Direct PDF link detected: {document_link}")
                        pdf_response = requests.get(
                            document_link, headers=request_headers, timeout=30
                        )
                        pdf_response.raise_for_status()

                        # Save the PDF
                        output_path = os.path.join(output_dir, f"{document_id}.pdf")
                        with open(output_path, "wb") as f:
                            f.write(pdf_response.content)

                        successful += 1
                        print(f"Successfully downloaded PDF: {document_id}.pdf")
                        success = True
                        break

                    # Special handling for officielebekendmakingen.nl
                    if "officielebekendmakingen.nl" in domain:
                        pdf_link = handle_officielebekendmakingen(
                            document_link, request_headers
                        )
                        if pdf_link:
                            print(
                                f"Found PDF link from officielebekendmakingen: {pdf_link}"
                            )
                            try:
                                pdf_headers = request_headers.copy()
                                pdf_headers["Referer"] = document_link

                                pdf_response = requests.get(
                                    pdf_link, headers=pdf_headers, timeout=30
                                )
                                pdf_response.raise_for_status()

                                output_path = os.path.join(
                                    output_dir, f"{document_id}.pdf"
                                )
                                with open(output_path, "wb") as f:
                                    f.write(pdf_response.content)

                                successful += 1
                                print(
                                    f"Successfully downloaded PDF from officielebekendmakingen: {document_id}.pdf"
                                )
                                success = True
                                break
                            except Exception as e:
                                print(
                                    f"Failed to download PDF from officielebekendmakingen: {str(e)}"
                                )
                                link_errors.append(
                                    f"Failed to download PDF from officielebekendmakingen: {str(e)}"
                                )
                                continue

                    # Special handling for tweedekamer.nl
                    if "tweedekamer.nl" in domain:
                        pdf_link = extract_pdf_from_tweedekamer(
                            document_link, request_headers
                        )
                        if pdf_link and pdf_link != "EXTRACT_TEXT":
                            print(f"Found PDF link from tweedekamer: {pdf_link}")
                            try:
                                pdf_response = requests.get(
                                    pdf_link, headers=request_headers, timeout=30
                                )
                                pdf_response.raise_for_status()

                                output_path = os.path.join(
                                    output_dir, f"{document_id}.pdf"
                                )
                                with open(output_path, "wb") as f:
                                    f.write(pdf_response.content)

                                successful += 1
                                print(
                                    f"Successfully downloaded PDF from tweedekamer: {document_id}.pdf"
                                )
                                success = True
                                break
                            except Exception as e:
                                print(
                                    f"Failed to download PDF from tweedekamer: {str(e)}"
                                )
                                continue
                        elif pdf_link == "EXTRACT_TEXT":
                            # Handle text extraction for parliamentary transcripts
                            response = requests.get(
                                document_link, headers=request_headers, timeout=30
                            )
                            response.raise_for_status()
                            soup = BeautifulSoup(response.text, "html.parser")

                            if content_div := soup.find("div", class_="content-block"):
                                # Extract and save as text file
                                text_content = content_div.get_text(strip=True)
                                output_path = os.path.join(
                                    output_dir, f"{document_id}.txt"
                                )
                                with open(output_path, "w", encoding="utf-8") as f:
                                    f.write(text_content)
                                successful += 1
                                print(
                                    f"Successfully saved text content: {document_id}.txt"
                                )
                                success = True
                                break

                    # Make the request with custom headers
                    response = requests.get(
                        document_link, headers=request_headers, timeout=30
                    )
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "").lower()

                    # If it's a PDF, save it directly
                    if "application/pdf" in content_type:
                        output_path = os.path.join(output_dir, f"{document_id}.pdf")
                        with open(output_path, "wb") as f:
                            f.write(response.content)

                        successful += 1
                        print(f"Successfully downloaded PDF: {document_id}.pdf")
                        success = True
                        break

                    # Handle HTML content - first try to find PDF links
                    if "text/html" in content_type:
                        html_content = response.text
                        soup = BeautifulSoup(html_content, "html.parser")

                        # Try to find PDF download links first
                        pdf_link = find_pdf_download_link(soup, document_link)
                        if pdf_link:
                            print(f"Found PDF download link: {pdf_link}")
                            try:
                                # Use custom headers for the PDF download
                                pdf_headers = request_headers.copy()
                                pdf_headers["Referer"] = document_link

                                pdf_response = requests.get(
                                    pdf_link, headers=pdf_headers, timeout=30
                                )
                                pdf_response.raise_for_status()

                                # Check if it's actually a PDF
                                if (
                                    "application/pdf"
                                    in pdf_response.headers.get(
                                        "Content-Type", ""
                                    ).lower()
                                ):
                                    output_path = os.path.join(
                                        output_dir, f"{document_id}.pdf"
                                    )
                                    with open(output_path, "wb") as f:
                                        f.write(pdf_response.content)

                                    successful += 1
                                    print(
                                        f"Successfully downloaded PDF from link: {document_id}.pdf"
                                    )
                                    success = True
                                    break
                            except Exception as e:
                                print(
                                    f"Failed to download PDF from link {pdf_link}: {str(e)}"
                                )
                                # Continue with text extraction as fallback

                        # If no PDF found, check content length and complexity
                        main_content = (
                            soup.find("article")
                            or soup.find("main")
                            or soup.find(
                                "div",
                                class_=lambda x: x
                                and ("content" in x.lower() or "article" in x.lower()),
                            )
                        )

                        if main_content:
                            # Check content length (text content over 1000 characters)
                            text_content = main_content.get_text(strip=True)
                            # Check for multiple paragraphs or sections
                            paragraphs = main_content.find_all(
                                ["p", "section", "div"],
                                class_=lambda x: x and "header" not in x.lower(),
                            )

                            if len(text_content) > 1000 and len(paragraphs) > 2:
                                print(
                                    "Significant content found, attempting PDF conversion"
                                )
                                try:
                                    output_path = os.path.join(
                                        output_dir, f"aaaa{document_id}.pdf"
                                    )
                                    website_to_pdf(document_link, output_path)

                                    # Verify the PDF was actually created and has content
                                    if (
                                        os.path.exists(output_path)
                                        and os.path.getsize(output_path) > 1000
                                    ):
                                        successful += 1
                                        print(
                                            f"Successfully saved webpage as PDF: {document_id}.pdf"
                                        )
                                        success = True
                                        break
                                    else:
                                        if os.path.exists(output_path):
                                            os.remove(
                                                output_path
                                            )  # Clean up empty/invalid PDF
                                        print(
                                            "PDF conversion failed: Output file empty or missing"
                                        )
                                        link_errors.append(
                                            "PDF conversion failed: Empty or missing output"
                                        )
                                except Exception as e:
                                    print(f"Failed to convert webpage to PDF: {str(e)}")
                                    link_errors.append(
                                        f"PDF conversion failed: {str(e)}"
                                    )
                                    # Continue with next link
                            else:
                                print(
                                    "Page content too short or simple, skipping PDF conversion"
                                )

                        # Check if the webpage is valid for conversion
                        if is_valid_webpage_for_conversion(soup, document_link):
                            print("Valid webpage content found, converting to PDF")
                            try:
                                output_path = os.path.join(
                                    output_dir, f"{document_id}.pdf"
                                )
                                website_to_pdf(document_link, output_path)
                                successful += 1
                                print(
                                    f"Successfully converted webpage to PDF: {document_id}.pdf"
                                )
                                success = True
                                break
                            except Exception as e:
                                print(f"Failed to convert webpage to PDF: {str(e)}")
                                link_errors.append(f"PDF conversion failed: {str(e)}")
                        else:
                            print("Page content not suitable for PDF conversion")
                            link_errors.append(
                                "Content not suitable for PDF conversion"
                            )

                    else:
                        # For other content types, save with appropriate extension
                        content = response.content
                        file_extension = determine_file_extension(
                            content_type, document_link
                        )

                        output_path = os.path.join(
                            output_dir, f"{document_id}.{file_extension}"
                        )
                        with open(output_path, "wb") as f:
                            f.write(content)

                        successful += 1
                        print(
                            f"Successfully downloaded: {document_id}.{file_extension}"
                        )
                        success = True
                        break

                except Exception as e:
                    error_msg = str(e)
                    print(f"Failed with link {document_link}: {error_msg}")
                    link_errors.append(error_msg)
                    # Continue to the next link

            if not success:
                failed += 1
                # Track failure by domain (using the last attempted domain)
                if "domain" in locals():
                    failures_by_domain[domain] = failures_by_domain.get(domain, 0) + 1
                print(f"All links failed for document at index {index}")

                # Add to failed links data
                failed_links_data.append(
                    {
                        "document_id": document_id,
                        "index": index,
                        "links": document_links_raw,
                        "errors": "; ".join(link_errors),
                    }
                )

        except Exception as e:
            failed += 1
            error_msg = str(e)
            # Track failure by domain
            domain = (
                extract_domain(document_links[0])
                if "document_links" in locals() and document_links
                else "unknown"
            )
            failures_by_domain[domain] = failures_by_domain.get(domain, 0) + 1
            print(f"Failed to process document at index {index}: {error_msg}")

            # Add to failed links data
            failed_links_data.append(
                {
                    "document_id": document_id,
                    "index": index,
                    "links": (
                        document_links_raw
                        if "document_links_raw" in locals()
                        else "N/A"
                    ),
                    "errors": error_msg,
                }
            )

    print(
        f"Download complete. Successful: {successful}, Failed: {failed}, Skipped: {skipped}, Excluded: {excluded}"
    )

    # Print summary of failures by domain
    if failures_by_domain:
        print("\nSummary of failures by domain:")
        for domain, count in sorted(
            failures_by_domain.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {domain}: {count} failures")

    # Save failed links to CSV
    if failed_links_data:
        failed_df = pd.DataFrame(failed_links_data)
        failed_df.to_csv(failed_links_file, index=False)
        print(f"\nSaved {len(failed_links_data)} failed links to {failed_links_file}")

    return successful, failed, skipped, excluded, failures_by_domain


def find_pdf_download_link(soup, url):
    """
    Find PDF download links in the page with enhanced detection

    Args:
        soup: BeautifulSoup object
        url: Original URL

    Returns:
        Absolute URL to PDF if found, None otherwise
    """
    domain = extract_domain(url)
    base_url = url

    # Common PDF link patterns - expanded for better detection
    pdf_link_patterns = [
        # Direct PDF links
        {
            "tag": "a",
            "attrs": {"href": lambda href: href and href.lower().endswith(".pdf")},
        },
        # Download buttons and links
        {
            "tag": "a",
            "attrs": {
                "class": lambda c: c
                and any(x in c.lower() for x in ["download", "btn-download", "pdf"])
            },
        },
        {
            "tag": "a",
            "attrs": {
                "title": lambda t: t and ("download" in t.lower() or "pdf" in t.lower())
            },
        },
        {
            "tag": "a",
            "attrs": {
                "aria-label": lambda t: t
                and ("download" in t.lower() or "pdf" in t.lower())
            },
        },
        # Links with download or PDF in text content
        {
            "tag": "a",
            "text": lambda t: t
            and (
                "download" in t.lower() or "pdf" in t.lower() or "document" in t.lower()
            ),
        },
        # Links with document icons
        {
            "tag": "a",
            "has_child": {
                "tag": "i",
                "class": lambda c: c
                and any(x in c.lower() for x in ["pdf", "file", "document"]),
            },
        },
        # Links with specific data attributes
        {"tag": "a", "attrs": {"data-format": "pdf"}},
        {"tag": "a", "attrs": {"data-filetype": "pdf"}},
    ]

    # Site-specific patterns
    if "rijksoverheid.nl" in domain or "overheid.nl" in domain:
        # Dutch government sites often have specific download buttons
        pdf_link_patterns.extend(
            [
                {
                    "tag": "a",
                    "attrs": {"class": lambda c: c and "download-button" in c},
                },
                {
                    "tag": "a",
                    "attrs": {"class": lambda c: c and "document-download" in c},
                },
                {"tag": "a", "attrs": {"data-decorator": "download"}},
                {"tag": "a", "attrs": {"class": "download"}},
            ]
        )

    # Try each pattern
    for pattern in pdf_link_patterns:
        if "has_child" in pattern:
            # Handle patterns with child elements
            parent_tag = pattern["tag"]
            child_spec = pattern["has_child"]

            for parent in soup.find_all(parent_tag):
                if parent.find(child_spec["tag"], class_=child_spec["class"]):
                    href = parent.get("href")
                    if href:
                        full_url = make_absolute_url(href, base_url)
                        if is_likely_pdf_link(full_url, parent.text):
                            return full_url
        elif "text" in pattern:
            # Handle patterns with text content check
            for link in soup.find_all(pattern["tag"]):
                if pattern["text"](link.get_text()):
                    href = link.get("href")
                    if href:
                        full_url = make_absolute_url(href, base_url)
                        if is_likely_pdf_link(full_url, link.text):
                            return full_url
        else:
            # Handle standard attribute patterns
            links = soup.find_all(pattern["tag"], attrs=pattern["attrs"])
            for link in links:
                href = link.get("href")
                if href:
                    full_url = make_absolute_url(href, base_url)
                    if is_likely_pdf_link(full_url, link.text):
                        return full_url

    # Special case for Dutch government sites with document viewers
    if "rijksoverheid.nl" in domain or "overheid.nl" in domain:
        # Look for document viewer with PDF data
        viewer_div = soup.find("div", {"class": "document-viewer"})
        if viewer_div:
            data_json = viewer_div.get("data-json")
            if data_json:
                try:
                    data = json.loads(data_json)
                    if "document" in data and "pdf" in data["document"]:
                        return make_absolute_url(data["document"]["pdf"], base_url)
                except Exception:
                    pass

    # Crawl one level deep to find PDF links on linked pages
    # This is useful for index pages that link to document pages
    if not url.lower().endswith(".pdf"):  # Avoid recursive crawling
        document_links = []

        # Look for links that might lead to document pages
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            link_text = link.get_text().lower()

            # Skip links that are clearly not document links
            if (
                not href
                or href.startswith("#")
                or href.startswith("javascript:")
                or "login" in href.lower()
                or "account" in href.lower()
            ):
                continue

            # Check if the link text suggests it might lead to a document
            if any(
                term in link_text
                for term in [
                    "document",
                    "pdf",
                    "download",
                    "view",
                    "rapport",
                    "report",
                    "paper",
                ]
            ) or any(
                term in href.lower()
                for term in ["doc", "pdf", "download", "view", "rapport", "report"]
            ):
                full_url = make_absolute_url(href, base_url)
                if (
                    full_url not in document_links
                    and extract_domain(full_url) == domain
                ):
                    document_links.append(full_url)

        # Limit to top 3 most promising links to avoid excessive crawling
        document_links = document_links[:3]

        # Try each potential document link
        for doc_link in document_links:
            try:
                print(f"Crawling potential document page: {doc_link}")
                doc_response = requests.get(doc_link, timeout=20)
                if doc_response.status_code == 200:
                    doc_soup = BeautifulSoup(doc_response.text, "html.parser")
                    pdf_link = find_pdf_download_link_simple(doc_soup, doc_link)
                    if pdf_link:
                        print(f"Found PDF link on secondary page: {pdf_link}")
                        return pdf_link
            except Exception as e:
                print(f"Error crawling {doc_link}: {str(e)}")
                continue

    return None


def find_pdf_download_link_simple(soup, url):
    """
    Simplified version of PDF link finder for secondary pages
    to avoid recursive crawling
    """
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if href:
            full_url = make_absolute_url(href, url)
            if full_url.lower().endswith(".pdf"):
                return full_url

            # Check for download links
            link_text = link.get_text().lower()
            if ("download" in link_text or "pdf" in link_text) and is_likely_pdf_link(
                full_url, link_text
            ):
                return full_url

    return None


def is_likely_pdf_link(url, link_text=""):
    """
    Check if a URL is likely to be a PDF link based on URL and link text

    Args:
        url: URL to check
        link_text: Text of the link (optional)

    Returns:
        bool: True if likely a PDF link
    """
    url_lower = url.lower()

    # Direct PDF extension
    if url_lower.endswith(".pdf"):
        return True

    # URL contains PDF indicators
    if any(
        term in url_lower
        for term in ["/pdf/", "pdf=", "format=pdf", "type=pdf", "download=pdf"]
    ):
        return True

    # Link text contains PDF indicators
    if link_text and any(
        term in link_text.lower()
        for term in ["pdf", "download document", "download report"]
    ):
        return True

    # Common document repository patterns
    if any(
        pattern in url_lower
        for pattern in ["/documents/", "/publications/", "/reports/", "/files/"]
    ):
        return True

    return False


def make_absolute_url(href, base_url):
    """Convert relative URL to absolute if needed"""
    if href.startswith("/") or not href.startswith(("http://", "https://")):
        from urllib.parse import urljoin

        return urljoin(base_url, href)
    return href


def determine_file_extension(content_type, url):
    """
    Determine file extension from content type or URL

    Args:
        content_type: Content-Type header
        url: URL of the document

    Returns:
        File extension (without dot)
    """
    # Default extension
    file_extension = "pdf"

    # Document formats
    if "application/pdf" in content_type:
        file_extension = "pdf"
    elif "application/msword" in content_type:
        file_extension = "doc"
    elif (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in content_type
    ):
        file_extension = "docx"
    elif "application/vnd.ms-excel" in content_type:
        file_extension = "xls"
    elif (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        in content_type
    ):
        file_extension = "xlsx"
    elif "application/vnd.ms-powerpoint" in content_type:
        file_extension = "ppt"
    elif (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        in content_type
    ):
        file_extension = "pptx"

    # Text formats
    elif "text/plain" in content_type:
        file_extension = "txt"
    elif "text/csv" in content_type:
        file_extension = "csv"
    elif "application/json" in content_type:
        file_extension = "json"
    elif "application/xml" in content_type or "text/xml" in content_type:
        file_extension = "xml"

    # Image formats
    elif "image/jpeg" in content_type:
        file_extension = "jpg"
    elif "image/png" in content_type:
        file_extension = "png"
    elif "image/gif" in content_type:
        file_extension = "gif"
    elif "image/tiff" in content_type:
        file_extension = "tiff"

    # Archive formats
    elif "application/zip" in content_type:
        file_extension = "zip"
    elif "application/x-rar-compressed" in content_type:
        file_extension = "rar"
    elif "application/gzip" in content_type:
        file_extension = "gz"

    # Fallback: try to extract extension from URL
    else:
        if url and "." in url:
            url_extension = url.split(".")[-1].lower()
            # Check if it's a reasonable extension (not too long)
            if len(url_extension) < 6:
                file_extension = url_extension

    return file_extension


def extract_domain(url):
    """
    Extract domain from URL

    Args:
        url: URL string

    Returns:
        Domain name
    """
    from urllib.parse import urlparse

    try:
        parsed_uri = urlparse(url)
        domain = "{uri.netloc}".format(uri=parsed_uri)
        # Remove www. prefix if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


# Excel URL mapping configuration
excel_url_mapping = [
    {
        "base_url": "https://open.minvws.nl/dossier/VWS-WOO/3773352-1061613-pdo",
        "csv_file": "data/metadata/Publicatierapport de inkoop van persoonlijke beschermingsmiddelen.xlsx",
    },
    {
        "base_url": "https://open.minvws.nl/dossier/VWS-WOO/3575317-1046458-pdo",
        "csv_file": "data/metadata/Publicatierapport chatberichten groepschat directeur communicatie.xlsx",
    },
    {
        "base_url": "https://open.minvws.nl/dossier/VWS-WOO/3814289-1065108-pdo",
        "csv_file": "data/metadata/Publicatierapport COVID-19 Campagnes.xlsx",
    },
    {
        "base_url": "https://open.minvws.nl/dossier/VWS-WOO/3574524-1046390-PDO",
        "csv_file": "data/metadata/Publicatierapport Overleg VWS april 2021.xlsx",
    },
]


def is_valid_webpage_for_conversion(soup, url):
    """
    Check if a webpage is valid for PDF conversion by analyzing its content and structure.

    Args:
        soup: BeautifulSoup object of the webpage
        url: URL of the webpage

    Returns:
        bool: True if the webpage should be converted to PDF
    """
    # Skip known non-content pages
    skip_patterns = ["login", "search", "contact", "404", "error", "not found"]
    if any(pattern in url.lower() for pattern in skip_patterns):
        return False

    # Check for main content areas with more inclusive terms
    content_areas = soup.find_all(
        ["article", "main", "div"],
        class_=lambda x: x
        and any(
            term in str(x).lower()
            for term in [
                "content",
                "article",
                "main",
                "body",
                "text",
                "detail",
                "document",
                "kamerstuk",
            ]
        ),
    )

    if not content_areas:
        return False

    # Analyze the main content
    main_content = max(
        content_areas, key=lambda x: len(x.get_text(strip=True)), default=None
    )
    if not main_content:
        return False

    text_content = main_content.get_text(strip=True)

    # More lenient content quality checks
    min_text_length = 200  # Reduced from 500
    min_paragraphs = 1  # Reduced from 3
    min_words_per_para = 10  # Reduced from 20

    # Count paragraphs with substantial content
    paragraphs = [
        p.get_text(strip=True) for p in main_content.find_all(["p", "div", "section"])
    ]
    substantial_paragraphs = [
        p for p in paragraphs if len(p.split()) >= min_words_per_para
    ]

    # Content structure checks
    has_headings = bool(main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
    has_lists = bool(main_content.find_all(["ul", "ol"]))
    has_tables = bool(main_content.find_all("table"))
    has_links = bool(main_content.find_all("a", href=True))
    has_downloads = bool(
        soup.find_all(
            "a", class_="m-button", href=lambda x: x and "downloads/document" in x
        )
    )

    # Additional checks for government websites
    gov_domains = ["overheid.nl", "rijksoverheid.nl", "tweedekamer.nl", "europa.eu"]
    is_gov_site = any(domain in url.lower() for domain in gov_domains)

    if is_gov_site:
        # Very lenient criteria for government sites
        return (
            len(text_content) >= 100  # Minimal text length requirement
            or has_tables
            or has_lists
            or has_downloads  # Include pages with download buttons
            or has_headings  # Include pages with any structure
            or has_links  # Include pages with links
        )

    # Still somewhat strict for non-government sites
    valid_content = (
        len(text_content) >= min_text_length
        and len(substantial_paragraphs) >= min_paragraphs
        and (has_headings or has_lists or has_tables or has_links)
    )

    return valid_content


if __name__ == "__main__":
    # Initialize metadata manager and get dataframe
    metadata_manager = MetadataManager(excel_url_mapping=excel_url_mapping)
    metadata_manager.metadata_df = metadata_manager.combine_duplicate_columns(
        metadata_manager.metadata_df
    )
    df = metadata_manager.metadata_df

    print(f"Loaded metadata with {len(df)} documents")
    print(f"Columns: {df.columns.tolist()}")

    # Check if any of the link columns exist
    link_columns = ["Publieke Link", "Publieke link"]
    has_link_column = any(col in df.columns for col in link_columns)

    df = df[df["Publieke Link"].notna() | df["Publieke link"].notna()]
    # Scrape documents
    successful, failed, skipped, excluded, failures_by_domain = scrape_public_links(
        df, "data/scraped_documents"
    )

    # Print summary of failures by domain
    if failures_by_domain:
        print("\nSummary of failures by domain:")
        for domain, count in sorted(
            failures_by_domain.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {domain}: {count} failures")
