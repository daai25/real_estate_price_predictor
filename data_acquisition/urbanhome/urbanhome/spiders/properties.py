import scrapy
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager
import time


class UrbanHomeSeleniumSpider(scrapy.Spider):
    name = 'urbanhome_selenium'
    start_urls = [
        'https://www.urbanhome.ch/suchen/mieten/wohnen',
        'https://www.urbanhome.ch/suchen/mieten/wohnung',
        'https://www.urbanhome.ch/suchen/mieten/haus',
        'https://www.urbanhome.ch/suchen/mieten/buero',
        'https://www.urbanhome.ch/suchen/mieten/gewerbe',
    ]
    allowed_domains = ['urbanhome.ch']

    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    def parse(self, response):
        self.driver.get(response.url)

        # Scroll to load more items (simulate infinite scrolling)
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for _ in range(10):  # Scroll 10 times
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        cards = soup.select('a.listing')

        for card in cards:
            title = card.select_one('p.card-text strong')
            price_text = card.select_one('h3.card-title')
            text_blocks = card.select('p.card-text')
            address_block = text_blocks[1] if len(text_blocks) > 1 else None

            title_text = title.get_text(strip=True) if title else None
            price = None
            if price_text:
                price_match = re.search(r"([\d'’]+)", price_text.get_text())
                if price_match:
                    price = int(price_match.group(1).replace("'", "").replace("’", ""))

            address, zip_code, city, region = [None] * 4
            if address_block:
                html = address_block.decode_contents()
                parts = re.split(r'<br\s*/?>', html)
                parts = [BeautifulSoup(p, 'html.parser').get_text(strip=True) for p in parts if p.strip()]
                
                if len(parts) >= 2:
                    address = parts[0]
                    tokens = parts[1].split()
            
                    if len(tokens) >= 2:
                        zip_code = tokens[0]
                        region = tokens[-1]
                        city = " ".join(tokens[1:-1]) if len(tokens) > 2 else tokens[1]

            item = {
                "title": title_text,
                "address": address,
                "zip_code": zip_code,
                "city": city,
                "region": region,
                "price": price,
            }

            detail_url = response.urljoin(card['href'])
            yield scrapy.Request(
                url=detail_url,
                callback=self.parse_detail,
                cb_kwargs={'item': item},
                dont_filter=True
            )

    def parse_detail(self, response, item):
        self.driver.get(response.url)
        time.sleep(2)  # Wait for JavaScript to render

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        def extract_value(label_text):
            for li in soup.select("li.list-group-item"):
                if label_text.lower() in li.text.lower():
                    value = li.select_one("span.value")
                    return value.get_text(strip=True) if value else None

            for dt in soup.select("dt"):
                if label_text.lower() in dt.get_text(strip=True).lower():
                    dd = dt.find_next_sibling("dd")
                    return dd.get_text(strip=True) if dd else None

            return None

        item['rooms'] = extract_value("Zimmer")
        item['area_sqm'] = extract_value("Fläche")

        floor = soup.select_one("dt.fst-normal:contains('Etage') + dd.col")
        item['floor'] = floor.get_text(strip=True) if floor else None

        item['availability_date'] = extract_value("verfügbar")

        item['has_balcony'] = False
        for dt in soup.select("dt.fst-normal"):
            if "balkon" in dt.get_text(strip=True).lower():
                dd = dt.find_next_sibling("dd")
                if dd and dd.select_one("svg"):
                    item['has_balcony'] = True
                    break

        desc_div = soup.select_one("div.description")
        item['description'] = desc_div.get_text(strip=True) if desc_div else None
        
        image_elements = soup.select("div.swiper-slide img")
        image_urls = []

        for img in image_elements:
            src = img.get("src")
            if src:
                full_url = response.urljoin(src)
                if full_url not in image_urls:
                    image_urls.append(full_url)

        item['image_urls'] = image_urls

        yield item

    def closed(self, reason):
        self.driver.quit()
