import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def handle_openkamer_site(url, headers):
    """
    Handle openkamer.org and similar sites to find the source document PDF

    Args:
        url: URL of the openkamer page
        headers: Request headers

    Returns:
        URL to PDF if found, None otherwise
    """
    try:
        print(f"Fetching openkamer page: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for "Bron antwoord" link
        bron_antwoord = None

        # Method 1: Look for table with Bron antwoord
        for th in soup.find_all("th"):
            if "Bron antwoord" in th.text:
                # Get the next td element
                td = th.find_next("td")
                if td and td.find("a"):
                    bron_antwoord = td.find("a")["href"]
                    print(f"Found Bron antwoord link: {bron_antwoord}")
                    break

        # Method 2: Look for links with specific text
        if not bron_antwoord:
            for link in soup.find_all("a"):
                if link.text and "bron antwoord" in link.text.lower():
                    bron_antwoord = link["href"]
                    print(f"Found Bron antwoord link (method 2): {bron_antwoord}")
                    break

        if bron_antwoord:
            # If the link is to officielebekendmakingen.nl, follow it to get the PDF
            if "officielebekendmakingen.nl" in bron_antwoord:
                return handle_officielebekendmakingen(bron_antwoord, headers)

            # Otherwise, return the link directly
            return bron_antwoord

        return None
    except Exception as e:
        print(f"Error handling openkamer site: {str(e)}")
        return None


def handle_officielebekendmakingen(url, headers):
    """
    Handle officielebekendmakingen.nl to find the PDF download link

    Args:
        url: URL of the officielebekendmakingen page
        headers: Request headers

    Returns:
        URL to PDF if found, None otherwise
    """
    try:
        # Direct PDF conversion for known URL patterns
        if "/ah-tk-" in url or "/kst-" in url or "/kv-" in url:
            # Replace .html with .pdf in the URL
            pdf_url = url.replace(".html", ".pdf")

            # Verify the PDF exists
            try:
                head_response = requests.head(pdf_url, headers=headers, timeout=10)
                if (
                    head_response.status_code == 200
                    and "application/pdf"
                    in head_response.headers.get("Content-Type", "")
                ):
                    return pdf_url
            except Exception as e:
                print(f"Error verifying direct PDF URL: {str(e)}")

        # If direct conversion fails, try existing methods
        print(f"Fetching officielebekendmakingen page: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for PDF download link in the page
        pdf_link = None

        # Method 1: Look for "authentieke versie" link
        auth_links = soup.find_all(
            "a", text=lambda t: t and "authentieke versie" in t.lower()
        )
        if auth_links:
            for link in auth_links:
                href = link.get("href")
                if href:
                    pdf_link = urljoin(url, href)
                    print(f"Found authentieke versie PDF link: {pdf_link}")
                    break

        # Method 2: Look for specific document patterns in the URL
        if not pdf_link:
            doc_patterns = {
                "ah-tk-": "aanhangsel",
                "kst-": "kamerstuk",
                "kv-": "kamervragen",
            }

            for pattern, doc_type in doc_patterns.items():
                match = re.search(f"/{pattern}([^/]+)\.html", url)
                if match:
                    doc_id = match.group(1)
                    pdf_link = (
                        f"https://zoek.officielebekendmakingen.nl/{pattern}{doc_id}.pdf"
                    )
                    print(f"Constructed PDF link for {doc_type}: {pdf_link}")
                    break
        if not pdf_link:
            url = url.replace(".html", ".pdf")
            pdf_link = url
        if pdf_link:
            # Verify the PDF link works
            try:
                pdf_headers = headers.copy()
                pdf_headers["Referer"] = url

                head_response = requests.head(pdf_link, headers=pdf_headers, timeout=10)
                if head_response.status_code == 200:
                    return pdf_link
                else:
                    print(
                        f"PDF link verification failed with status {head_response.status_code}"
                    )
            except Exception as e:
                print(f"Error verifying PDF link: {str(e)}")
                # Return the link anyway as a last resort
                return pdf_link

        return None
    except Exception as e:
        print(f"Error handling officielebekendmakingen: {str(e)}")
        return None


def extract_pdf_from_tweedekamer(url, headers):
    """
    Extract PDF links from tweedekamer.nl website
    """
    try:
        # If URL contains 'downloads/document', it's a direct download link
        if "/downloads/document" in url:
            return url  # Return immediately for direct download links

        session = requests.Session()
        response = session.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")

        # First priority: Look for m-button download links
        download_links = soup.find_all(
            "a", class_="m-button", href=lambda x: x and "downloads/document" in x
        )
        if download_links:
            href = download_links[0].get("href")
            if href.startswith("/"):
                return urljoin(url, href)
            return href

        # Method 2: Look for repository-style PDF links
        if not download_links:
            download_links = soup.find_all(
                "a",
                href=lambda x: x
                and (
                    x.endswith(".pdf")
                    or "pdf" in x.lower()
                    or "bitstream" in x.lower()
                    or "download" in x.lower()
                ),
            )

        for link in download_links:
            href = link.get("href", "")
            if href.startswith("/"):
                full_url = urljoin(url, href)
            else:
                full_url = href

            try:
                head_response = session.head(full_url, headers=headers, timeout=10)
                if head_response.status_code == 200:
                    content_type = head_response.headers.get("Content-Type", "").lower()
                    if (
                        "application/pdf" in content_type
                        or "application/octet-stream" in content_type
                        or full_url.lower().endswith(".pdf")
                    ):
                        return full_url
            except Exception as e:
                print(f"Error checking download link {full_url}: {str(e)}")
                continue

        # Method 3: Look for document ID in the URL and construct download link
        if "id=" in url:
            doc_id = url.split("id=")[-1].split("&")[0]
            download_url = f"https://www.tweedekamer.nl/downloads/document?id={doc_id}"
            try:
                head_response = session.head(download_url, headers=headers, timeout=10)
                if head_response.status_code == 200:
                    content_type = head_response.headers.get("Content-Type", "").lower()
                    if (
                        "application/pdf" in content_type
                        or "application/octet-stream" in content_type
                    ):
                        return download_url
            except Exception as e:
                print(
                    f"Error checking constructed download URL {download_url}: {str(e)}"
                )

        # Look for PDF download links
        pdf_links = []
        pdf_links.append(url)
        # Method 1: Look for direct PDF links and download links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                href.lower().endswith(".pdf")
                or "download" in href.lower()
                and ("document" in href.lower() or "verslag" in href.lower())
            ):
                # Verify if it's a PDF by checking headers first
                full_url = urljoin(url, href)
                try:
                    head_response = session.head(full_url, headers=headers, timeout=10)
                    content_type = head_response.headers.get("Content-Type", "").lower()
                    if head_response.status_code == 200 and (
                        "application/pdf" in content_type
                        or "application/octet-stream" in content_type
                    ):
                        return full_url
                except Exception as e:
                    print(f"Error checking link {full_url}: {str(e)}")
                pdf_links.append(full_url)

        # Method 2: Look for document ID and construct PDF URL
        document_id = None
        for meta in soup.find_all("meta", property="og:url"):
            content = meta.get("content", "")
            if "/detail/" in content:
                parts = content.split("/")
                if len(parts) > 0:
                    document_id = parts[-1]

        if document_id:
            # Construct potential PDF URLs based on document ID
            pdf_links.append(
                f"https://www.tweedekamer.nl/downloads/document?id={document_id}"
            )

        # Modified verification logic for all potential PDF links
        for pdf_link in pdf_links:
            try:
                head_response = session.head(pdf_link, headers=headers, timeout=10)
                if head_response.status_code == 200:
                    content_type = head_response.headers.get("Content-Type", "").lower()
                    if (
                        "application/pdf" in content_type
                        or "application/octet-stream" in content_type
                    ):
                        return pdf_link
                    elif "text/html" in content_type:
                        # This might be a download page, try to get the actual PDF
                        get_response = session.get(
                            pdf_link, headers=headers, timeout=30
                        )
                        if get_response.status_code == 200:
                            soup = BeautifulSoup(get_response.text, "html.parser")
                            for a in soup.find_all("a", href=True):
                                href = a["href"]
                                full_url = urljoin(pdf_link, href)
                                try:
                                    head_check = session.head(
                                        full_url, headers=headers, timeout=10
                                    )
                                    if head_check.status_code == 200 and (
                                        "application/pdf"
                                        in head_check.headers.get(
                                            "Content-Type", ""
                                        ).lower()
                                        or "application/octet-stream"
                                        in head_check.headers.get(
                                            "Content-Type", ""
                                        ).lower()
                                    ):
                                        return full_url
                                except Exception as e:
                                    print(
                                        f"Error checking nested link {full_url}: {str(e)}"
                                    )
                                    continue
            except Exception as e:
                print(f"Error checking PDF link {pdf_link}: {str(e)}")
                continue

        # If no PDF links found, we'll need to scrape the content and save as text
        # This is a fallback for parliamentary transcripts that don't have PDF versions
        if soup.find("div", class_="content-block"):
            return "EXTRACT_TEXT"

        return None
    except Exception as e:
        print(f"Error extracting PDF from tweedekamer.nl: {str(e)}")
        return None


def website_to_pdf(url, output_pdf_path):
    """
    Convert a website to a PDF file using Playwright for accurate rendering.

    Args:
        url (str): The URL of the website to convert.
        output_pdf_path (str): The path to save the PDF file.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        )
        page = context.new_page()

        try:
            # Navigate with extended timeout
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for any of these common content selectors
            content_selectors = [
                "article",
                "main",
                ".content",
                ".article",
                ".post",
                ".reader-content",
                "#content",
                "#main",
            ]

            # Try to wait for at least one content selector
            for selector in content_selectors:
                try:
                    page.wait_for_selector(selector, timeout=5000)
                    break
                except Exception:
                    continue

            # Additional wait for dynamic content
            page.wait_for_timeout(2000)

            # Fix letter spacing issues and remove overlays
            page.evaluate(
                """() => {
                // Remove common overlay elements
                const selectors = [
                    '.modal', '.popup', '.overlay', 
                    '[class*="cookie"]', '[class*="consent"]',
                    '[class*="banner"]', '[class*="notification"]',
                    '[class*="subscribe"]', '[class*="newsletter"]'
                ];
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
                
                // Fix letter spacing issues
                const styleElement = document.createElement('style');
                styleElement.textContent = `
                    * {
                        letter-spacing: normal !important;
                        word-spacing: normal !important;
                        font-kerning: normal !important;
                        text-rendering: optimizeLegibility !important;
                        font-family: Arial, Helvetica, sans-serif !important;
                    }
                    p, span, div, td, th, li, a {
                        letter-spacing: normal !important;
                        word-spacing: normal !important;
                    }
                    table {
                        border-collapse: collapse !important;
                    }
                    td, th {
                        padding: 4px !important;
                    }
                `;
                document.head.appendChild(styleElement);
                
                // Fix specific issues with rijksfinancien.nl
                if (window.location.hostname.includes('rijksfinancien.nl')) {
                    // Remove any problematic scripts or styles
                    document.querySelectorAll('script[src*="piwik"]').forEach(el => el.remove());
                    
                    // Fix any specific elements with spacing issues
                    document.querySelectorAll('.table-container').forEach(table => {
                        table.style.width = '100%';
                        table.style.maxWidth = '100%';
                        table.style.overflowX = 'visible';
                    });
                }
            }"""
            )

            # Generate PDF with adjusted settings
            page.pdf(
                path=output_pdf_path,
                format="A4",
                margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"},
                print_background=True,
                scale=0.95,  # Slightly increased scale
                prefer_css_page_size=False,  # Changed to false to ensure our settings take precedence
            )

            # Verify the PDF was created successfully
            if (
                not os.path.exists(output_pdf_path)
                or os.path.getsize(output_pdf_path) < 1000
            ):
                raise Exception("Generated PDF is empty or too small")

        except Exception as e:
            if os.path.exists(output_pdf_path):
                os.remove(output_pdf_path)  # Clean up any partial/invalid PDF
            raise Exception(f"Failed to convert webpage to PDF: {str(e)}")

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    url = (
        "https://www.tweedekamer.nl/kamerstukken/plenaire_verslagen/detail/2020-2021/84"
    )
    output_pdf_path = "output.pdf"
    website_to_pdf(url, output_pdf_path)
    print(f"PDF saved to {output_pdf_path}")
