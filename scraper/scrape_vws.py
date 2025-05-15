import logging
import os
import random
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


class WobWooScraper:
    def __init__(self, base_url, output_dir="scraped_woo", redownload=True, test=False):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Add redownload flag
        self.redownload = redownload
        self.test = test

        self.test_besluiten = [
            "https://open.minvws.nl/dossier/VWS-WOO/3342474-1026791-wjz"
        ]  # ,"https://open.minvws.nl/dossier/VWS-WOO/3939612-1069713-pdo","https://open.minvws.nl/dossier/VWS-WC/001", "https://open.minvws.nl/dossier/VWS-WOO/3752957-1059996-pdo", "https://open.minvws.nl/dossier/VWS-WC/142", "https://open.minvws.nl/dossier/VWS-WC/020", "https://open.minvws.nl/dossier/VWS-WOO/1846768-219962-wjz"]
        # Create file handler
        log_file = os.path.join(output_dir, "scraper.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.base_url = base_url
        self.output_dir = output_dir
        self.documents_folder = f"{output_dir}/documents"
        self.inventory_folder = f"{output_dir}/inventory"
        self.besluit_folder = f"{output_dir}/besluiten"

        os.makedirs(self.documents_folder, exist_ok=True)
        os.makedirs(self.inventory_folder, exist_ok=True)
        os.makedirs(self.besluit_folder, exist_ok=True)

        self.session = requests.Session()

        # Split metadata into two separate dataframes
        self.besluiten_df = pd.DataFrame(
            columns=["besluit_id", "besluit_title", "besluit_url", "doc_count"]
        )

        self.documents_df = pd.DataFrame(
            columns=[
                "besluit_id",
                "document_id",
                "document_name",
                "document_date",
                "document_url",
                "document_type",
                "openness_status",
                "file_path",
                "file_size",
                "page_count",
                "file_type",
                "download_url",
                "inventory_number",
            ]
        )

        # Add a user agent to avoid being blocked
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        self.logger.info(f"Scraper initialized with base URL: {base_url}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Redownload flag: {redownload}")

    def scrape_starting_page(self, url):
        """Scrape all pages starting from the initial URL"""
        self.logger.info(f"Starting pagination from: {url}")
        all_besluiten = []
        base_url = url.split("?")[0]
        params = dict(
            pair.split("=") for pair in url.split("?")[1].split("&") if "=" in pair
        )

        # Get the first page to determine total pages
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all page number links
        page_links = soup.find_all(
            "a", {"data-e2e-name": lambda x: x and x.startswith("page-number-")}
        )
        if page_links:
            # Get the highest page number
            max_page = max(
                int(link["data-e2e-name"].split("-")[-1]) for link in page_links
            )
            self.logger.info(f"Found {max_page} total pages to scrape")
        else:
            max_page = 1
            self.logger.info("No pagination found, only processing first page")

        # Iterate through all pages
        for current_page in range(1, max_page + 1):
            # Update page parameter
            params["page"] = str(current_page)
            page_url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}#search-results"

            self.logger.info(f"Scraping page {current_page} of {max_page}: {page_url}")

            # Get page content
            response = self.session.get(page_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all search results on current page
            search_results = soup.find_all(
                "li", {"class": "woo-search-result", "data-e2e-name": "search-result"}
            )

            if not search_results:
                self.logger.warning(f"No results found on page {current_page}")
                continue

            page_besluiten = []
            for result in search_results:
                # Check if there are documents
                doc_count_elem = result.find("li", {"data-e2e-name": "nr-of-documents"})
                if not doc_count_elem:
                    continue

                doc_count_text = doc_count_elem.get_text(strip=True)
                doc_count = (
                    int(re.search(r"(\d+)", doc_count_text).group(1))
                    if re.search(r"(\d+)", doc_count_text)
                    else 0
                )

                # Get the link to the besluit
                link_elem = result.find("a", {"data-e2e-name": "main-link"})
                if link_elem and "href" in link_elem.attrs:
                    besluit_url = urljoin(self.base_url, link_elem["href"])
                    besluit_title = link_elem.get_text(strip=True)
                    besluit_id = link_elem["href"].split("/")[-1]

                    page_besluiten.append(
                        {
                            "url": besluit_url,
                            "title": besluit_title,
                            "id": besluit_id,
                            "doc_count": doc_count,
                        }
                    )

                    # Add to besluiten dataframe
                    self.besluiten_df = pd.concat(
                        [
                            self.besluiten_df,
                            pd.DataFrame(
                                [
                                    {
                                        "besluit_id": besluit_id,
                                        "besluit_title": besluit_title,
                                        "besluit_url": besluit_url,
                                        "doc_count": doc_count,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

            if page_besluiten:
                self.logger.info(
                    f"Found {len(page_besluiten)} besluiten on page {current_page}"
                )
                all_besluiten.extend(page_besluiten)
            else:
                self.logger.warning(f"No valid besluiten found on page {current_page}")

            # Add delay between pages
            if current_page < max_page:
                delay = random.uniform(2, 4)
                self.logger.debug(f"Waiting {delay:.2f} seconds before next page")
                time.sleep(delay)

        self.logger.info(
            f"Total besluiten found across all {max_page} pages: {len(all_besluiten)}"
        )
        return all_besluiten

    def get_inventory_and_besluit_brief_id(self, soup, besluit_info, documents):
        """Download inventory and besluit brief files"""
        # First download and process the inventarislijst for metadata
        inventory_link = soup.find(
            "a", {"data-e2e-name": "download-inventory-file-link"}
        )
        if inventory_link and "href" in inventory_link.attrs:
            inventory_url = urljoin(self.base_url, inventory_link["href"])
            inventory_filename = f"inventarislijst_{besluit_info['id']}.xlsx"
            inventory_path = os.path.join(self.inventory_folder, inventory_filename)
            self.download_file(inventory_url, inventory_path)

            # Add inventory to documents list
            documents.append(
                {
                    "besluit_id": besluit_info["id"],
                    "document_id": f"{besluit_info['id']}-inv",
                    "document_name": inventory_filename,
                    "document_date": None,
                    "document_url": inventory_url,
                    "document_type": "inventory",
                    "openness_status": "inventory",
                    "inventory_number": "INV",
                }
            )

        # Download the besluitbrief
        besluitbrief_link = soup.find(
            "a", {"data-e2e-name": "main-document-detail-link"}
        )
        if besluitbrief_link and "href" in besluitbrief_link.attrs:
            detail_url = urljoin(self.base_url, besluitbrief_link["href"])
            # besluitbrief_text = besluitbrief_link.get_text(strip=True)

            try:
                detail_response = self.session.get(detail_url)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.text, "html.parser")

                download_link = detail_soup.find(
                    "a", {"data-e2e-name": "download-file-link"}
                )

                if download_link and "href" in download_link.attrs:
                    besluitbrief_url = urljoin(self.base_url, download_link["href"])
                    besluitbrief_filename = f"besluitbrief_{besluit_info['id']}.pdf"
                    besluitbrief_path = os.path.join(
                        self.besluit_folder, besluitbrief_filename
                    )
                    self.download_file(besluitbrief_url, besluitbrief_path)

                    # Add besluitbrief to documents list
                    documents.append(
                        {
                            "besluit_id": besluit_info["id"],
                            "document_id": f"{besluit_info['id']}-brief",
                            "document_name": besluitbrief_filename,
                            "document_date": None,
                            "document_url": besluitbrief_url,
                            "document_type": "besluitbrief",
                            "openness_status": "besluitbrief",
                            "inventory_number": "BRIEF",
                        }
                    )
                else:
                    self.logger.error(
                        f"Could not find download link for besluitbrief on detail page: {detail_url}"
                    )

            except Exception as e:
                self.logger.error(f"Error fetching besluitbrief detail page: {e}")

        return documents

    def scrape_besluit_page(self, besluit_info):
        """Scrape a specific besluit page to find documents"""
        documents = []
        self.logger.info(
            f"Scraping besluit page: {besluit_info['url']} - {besluit_info['title']}"
        )

        response = self.session.get(besluit_info["url"])
        soup = BeautifulSoup(response.text, "html.parser")

        documents = self.get_inventory_and_besluit_brief_id(
            soup, besluit_info, documents
        )

        # Find all woo-tables
        woo_tables = soup.find_all("table", {"class": "woo-table"})

        # Process metadata table first
        metadata_table = next(
            (
                table
                for table in woo_tables
                if table.find_previous("h2", {"class": "woo-h2"})
                and table.find_previous("h2", {"class": "woo-h2"}).get_text(strip=True)
                == "Over dit besluit"
            ),
            None,
        )
        if metadata_table:
            self.get_besluit_metadata(metadata_table, besluit_info)
            # Download the besluit brief

        # Process document sections with pagination
        sections = [
            ("(Deels) openbaar gemaakt", "deels_openbaar"),
            ("Reeds openbaar", "reeds_openbaar"),
        ]

        found_sections = False
        for section_title, status in sections:
            self.logger.info(f"Processing section: {section_title}")
            # Find the section's pagination container
            section_header = soup.find("h2", string=lambda x: x and section_title in x)
            if not section_header:
                continue

            found_sections = True
            # Get the initial table for this section
            current_table = section_header.find_next("table", {"class": "woo-table"})
            if not current_table:
                continue

            # Extract documents from first page
            documents.extend(
                self.extract_documents_from_table(current_table, besluit_info, status)
            )

            # Find pagination for this section
            pagination = section_header.find_next("nav", {"class": "pagination"})
            if pagination:
                # Find all page links
                page_links = pagination.find_all(
                    "a", {"data-e2e-name": lambda x: x and x.startswith("page-number-")}
                )
                if page_links:
                    max_page = max(
                        int(link["data-e2e-name"].split("-")[-1]) for link in page_links
                    )

                    # Process remaining pages
                    for page in range(2, max_page + 1):
                        self.logger.info(
                            f"Processing {section_title} page {page} of {max_page}"
                        )

                        # Construct page URL
                        if status == "deels_openbaar":
                            page_url = f"{besluit_info['url']}?pu={page}#tabcontrol-1"
                        elif status == "reeds_openbaar":
                            page_url = f"{besluit_info['url']}?pu={page}#tabcontrol-2"

                        response = self.session.get(page_url)
                        page_soup = BeautifulSoup(response.text, "html.parser")

                        # Find the table in the paginated response
                        page_table = page_soup.find("table", {"class": "woo-table"})
                        if page_table:
                            documents.extend(
                                self.extract_documents_from_table(
                                    page_table, besluit_info, status
                                )
                            )

                        # Add delay between pages
                        time.sleep(random.uniform(1, 2))

        # If no sections were found with headers, look for a single table of deels openbaar documents
        if not found_sections:
            # Find the first woo-table that's not the metadata table
            doc_table = next(
                (table for table in woo_tables if table != metadata_table), None
            )
            if doc_table:
                self.logger.info(
                    "Found single table of documents without section header"
                )
                documents.extend(
                    self.extract_documents_from_table(
                        doc_table, besluit_info, "deels_openbaar"
                    )
                )

                # Check for pagination without section
                pagination = soup.find("nav", {"class": "pagination"})
                if pagination:
                    page_links = pagination.find_all(
                        "a",
                        {"data-e2e-name": lambda x: x and x.startswith("page-number-")},
                    )
                    if page_links:
                        max_page = max(
                            int(link["data-e2e-name"].split("-")[-1])
                            for link in page_links
                        )

                        # Process remaining pages
                        for page in range(2, max_page + 1):
                            self.logger.info(f"Processing page {page} of {max_page}")

                            # Construct page URL without section parameter

                            page_url = f"{besluit_info['url']}?pu={page}#tabcontrol-1"

                            response = self.session.get(page_url)
                            page_soup = BeautifulSoup(response.text, "html.parser")

                            # Find the table in the paginated response
                            page_table = page_soup.find("table", {"class": "woo-table"})
                            if page_table:
                                documents.extend(
                                    self.extract_documents_from_table(
                                        page_table, besluit_info, "deels_openbaar"
                                    )
                                )

                            # Add delay between pages
                            time.sleep(random.uniform(1, 2))

        # Process each document
        for doc in documents:
            self.logger.info(f"Processing document: {doc['document_name']}")
            self.process_document(doc)

        # Save metadata after each besluit is processed
        self.save_metadata()

        return documents

    def extract_documents_from_table(self, table, besluit_info, status):
        """Extract documents from a woo-table"""
        documents = []
        rows = table.find_all("tr")

        for row in rows:
            if row.find("th"):  # Skip header row
                continue

            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            doc_number = cells[0].get_text(strip=True)
            doc_link = cells[2].find("a", {"class": "woo-a"})

            if not doc_link:
                continue

            doc_url = urljoin(self.base_url, doc_link["href"])
            doc_name = doc_link.get_text(strip=True)

            # Extract date if available
            doc_date = None
            date_cell = cells[3] if len(cells) > 3 else None
            if date_cell:
                date_elem = date_cell.find("time")
                if date_elem and "datetime" in date_elem.attrs:
                    doc_date = date_elem["datetime"]

            # Get document type from icon if available
            doc_type = "unknown"
            type_cell = cells[1] if len(cells) > 1 else None
            if type_cell:
                type_icon = type_cell.find("use")
                if type_icon and "xlink:href" in type_icon.attrs:
                    icon_ref = type_icon["xlink:href"]
                    doc_type = icon_ref.split("#")[-1]

            documents.append(
                {
                    "besluit_id": besluit_info["id"],
                    "besluit_title": besluit_info["title"],
                    "document_id": f"{besluit_info['id']}-{doc_number}",
                    "document_name": doc_name,
                    "document_date": doc_date,
                    "document_url": doc_url,
                    "document_type": doc_type,
                    "openness_status": status,
                    "inventory_number": doc_number,
                }
            )

        return documents

    def get_besluit_metadata(self, metadata_table, besluit_info):
        """Extract metadata from the 'Over dit besluit' table"""
        metadata = {}
        rows = metadata_table.find_all("tr")

        for row in rows:
            header = row.find("th")
            value = row.find("td")
            if header and value:
                key = header.get_text(strip=True)

                if key == "Soort besluit":
                    metadata["Soort besluit"] = value.get_text(strip=True)
                elif key == "Type besluit":
                    metadata["Type besluit"] = value.get_text(strip=True)
                elif key == "Verantwoordelijk(en)":
                    metadata["Verantwoordelijk(en)"] = value.get_text(strip=True)
                elif key == "Periode":
                    metadata["Periode"] = value.get_text(strip=True)
                elif key == "Datum besluit":
                    date_elem = value.find("time")
                    metadata["Datum besluit"] = (
                        date_elem["datetime"] if date_elem else None
                    )
                elif key == "Onderwerp":
                    subjects = value.find_all("a")
                    metadata["Onderwerp"] = [
                        subject.get_text(strip=True) for subject in subjects
                    ]
                elif key == "Omvang openbaarmaking":
                    doc_count = value.find(
                        "span", {"data-e2e-name": "dossier-document-count"}
                    )
                    if doc_count:
                        count_text = doc_count.get_text(strip=True)
                        metadata["Aantal documenten"] = int(
                            re.search(r"(\d+)", count_text).group(1)
                        )
                    # Extract page count if available
                    page_count_match = re.search(
                        r"(\d+)\s*pagina", value.get_text(strip=True)
                    )
                    if page_count_match:
                        metadata["Aantal pagina's"] = int(page_count_match.group(1))

        # Update besluiten_df with new metadata
        update_dict = {
            "besluit_id": besluit_info["id"],
            "besluit_title": besluit_info["title"],
            "besluit_url": besluit_info["url"],
            **metadata,
        }
        self.besluiten_df.loc[
            self.besluiten_df["besluit_id"] == besluit_info["id"], update_dict.keys()
        ] = update_dict.values()

    def process_document(self, doc_info):
        """Process a document page to extract metadata and download files"""
        self.logger.info(
            f"Processing document: {doc_info['document_name']} ({doc_info['document_url']})"
        )

        try:
            # Get the document page
            response = self.session.get(doc_info["document_url"])
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract additional metadata from the document page
            # Find document date if not already set
            if not doc_info.get("document_date"):
                date_element = soup.select_one("time")
                if date_element:
                    doc_info["document_date"] = date_element.get_text(strip=True)

            # Extract file size and page count for PDFs
            pdf_info = soup.select_one("p.pb-1.woo-muted.text-sm")
            if pdf_info:
                pdf_text = pdf_info.get_text(strip=True)
                size_match = re.search(r"(\d+(?:\.\d+)?\s*[KMG]B)", pdf_text)
                pages_match = re.search(r"(\d+)\s*pagina", pdf_text)

                if size_match:
                    doc_info["file_size"] = size_match.group(1)
                if pages_match:
                    doc_info["page_count"] = int(pages_match.group(1))

            # For deels_openbaar, look for download link
            if doc_info["openness_status"] == "deels_openbaar":
                download_link = soup.select_one("a[download]")
                if download_link:
                    download_url = urljoin(self.base_url, download_link["href"])

                    # Create a filename based on document name
                    filename = f"{doc_info['document_id']}.pdf"
                    file_path = os.path.join(self.documents_folder, filename)

                    # Download the file
                    self.download_file(download_url, file_path)

                    # Add to documents metadata instead of metadata_df
                    self.documents_df = pd.concat(
                        [
                            self.documents_df,
                            pd.DataFrame(
                                [
                                    {
                                        "besluit_id": doc_info["besluit_id"],
                                        "document_id": doc_info["document_id"],
                                        "document_name": doc_info["document_name"],
                                        "document_date": doc_info["document_date"],
                                        "document_url": doc_info["document_url"],
                                        "document_type": doc_info["document_type"],
                                        "openness_status": doc_info["openness_status"],
                                        "file_path": file_path,
                                        "file_type": "pdf",
                                        "file_size": doc_info.get("file_size"),
                                        "page_count": doc_info.get("page_count"),
                                        "download_url": download_url,
                                        "inventory_number": doc_info[
                                            "inventory_number"
                                        ],
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

            # For reeds_openbaar, check if there's a link in the document
            elif doc_info["openness_status"] == "reeds_openbaar":
                # Look for external links
                external_links = soup.find_all(
                    "a",
                    {
                        "class": "woo-a",
                        "href": lambda x: x
                        and (x.startswith("http") or x.startswith("www")),
                    },
                )

                if external_links:
                    for link in external_links:
                        link_url = link["href"]
                        # link_text = link.get_text(strip=True)

                        # Add to documents metadata instead of metadata_df
                        self.documents_df = pd.concat(
                            [
                                self.documents_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "besluit_id": doc_info["besluit_id"],
                                            "document_id": doc_info["document_id"],
                                            "document_name": doc_info["document_name"],
                                            "document_date": doc_info["document_date"],
                                            "document_url": doc_info["document_url"],
                                            "document_type": doc_info["document_type"],
                                            "openness_status": doc_info[
                                                "openness_status"
                                            ],
                                            "file_path": None,
                                            "file_type": None,
                                            "file_size": None,
                                            "page_count": None,
                                            "download_url": link_url,
                                            "inventory_number": doc_info[
                                                "inventory_number"
                                            ],
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )

        except Exception as e:
            self.logger.error(
                f"Error processing document {doc_info['document_name']}: {e}"
            )

    def download_file(self, url, path):
        """Download a file from URL to the specified path"""
        self.logger.info(f"Downloading: {url} to {path}")

        # Check if file already exists and redownload flag is False
        if os.path.exists(path) and not self.redownload:
            self.logger.info(f"File already exists: {path}")
            return

        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Downloaded: {path}")

            # Add a small delay to avoid overloading the server
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")

    def sanitize_filename(self, filename):
        """Sanitize a filename to be safe for all operating systems"""
        # Split filename into name and extension
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = ".pdf"  # Default to .pdf if no extension

        # Remove or replace invalid characters
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        # Remove leading/trailing spaces and dots
        name = name.strip(". ")
        # Replace multiple spaces with single underscore
        name = re.sub(r"\s+", "_", name)
        # Ensure filename is not too long (max 100 chars for the name part)
        # Subtract length of extension to ensure full filename including extension stays within limits
        max_name_length = 100 - len(ext)
        name = name[:max_name_length]

        # Combine name and extension
        return f"{name}{ext}"

    def save_metadata(self):
        """Save metadata to CSV files"""
        besluiten_path = os.path.join(self.output_dir, "besluiten_metadata.csv")
        documents_path = os.path.join(self.output_dir, "documents_metadata.csv")

        self.besluiten_df.to_csv(besluiten_path, index=False)
        self.documents_df.to_csv(documents_path, index=False)

        self.logger.debug(f"Metadata saved to: {besluiten_path} and {documents_path}")

    def run(self, starting_url):
        """Run the scraper starting from the given URL"""
        # Process all pages
        besluiten = self.scrape_starting_page(starting_url)
        self.logger.info(f"Found {len(besluiten)} total besluiten across all pages")

        # Save metadata after finding all besluiten
        self.save_metadata()

        # Process each besluit
        for besluit in besluiten:
            try:
                if self.test:
                    if besluit["url"] in self.test_besluiten:
                        self.logger.info(f"Processing besluit: {besluit['title']}")
                        self.scrape_besluit_page(besluit)
                else:
                    # If not in test mode, process all besluiten
                    self.logger.info(f"Processing besluit: {besluit['title']}")
                    self.scrape_besluit_page(besluit)

                # Add a delay between requests
                time.sleep(random.uniform(2, 5))
            except Exception as e:
                self.logger.error(f"Error processing besluit {besluit['title']}: {e}")
                continue

        # Final save of metadata (though it should already be up to date)
        self.save_metadata()

        self.logger.info("Scraping completed. Metadata saved to the output directory.")

        return self.besluiten_df, self.documents_df


if __name__ == "__main__":
    # Set up logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Initialize the scraper with redownload flag
    base_url = "https://open.minvws.nl"  # Replace with the actual base URL
    scraper = WobWooScraper(
        base_url, output_dir="data/woo_scraped", redownload=True, test=False
    )

    # Run the scraper
    starting_url = "https://open.minvws.nl/zoeken?sort=_score&sortorder=desc&doctype%5B%5D=dossier&doctype%5B%5D=dossier.publication"  # Replace with the actual starting page URL
    besluiten_df, documents_df = scraper.run(starting_url)

    # Display the results
    logger.info(f"Total besluiten scraped: {len(besluiten_df)}")
    logger.info(f"Total documents scraped: {len(documents_df)}")
    logger.debug("First few rows of besluiten metadata:")
    logger.debug(besluiten_df.head())
    logger.debug("First few rows of documents metadata:")
    logger.debug(documents_df.head())
