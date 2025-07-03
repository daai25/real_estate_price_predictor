import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from urllib.parse import quote
import re
import time


class FlatfoxSeleniumSpider(scrapy.Spider):
    name = 'flatfox_selenium'
    allowed_domains = ['flatfox.ch']
    SWISS_CITIES = [
        "Zurich", "Geneva", "Basel", "Bern", "Lausanne", "Lucerne", "St. Gallen", "Lugano",
        "Winterthur", "Biel/Bienne", "Thun", "Köniz", "La Chaux-de-Fonds", "Schaffhausen",
        "Fribourg", "Chur", "Neuchâtel", "Vernier", "Uster", "Sion"
        # You can add more cities here
    ]

    def __init__(self):
        options = Options()
        #options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def get_city_url(self, city_name):
        geolocator = Nominatim(user_agent="flatfox_scraper")
        location = geolocator.geocode(city_name, exactly_one=True, addressdetails=True, bounded=True)
        if not location:
            return None
        south, north, west, east = map(float, location.raw["boundingbox"])
        lat_range = north - south
        lon_range = east - west
        s = south - lat_range * 0.05
        n = north + lat_range * 0.05
        w = west - lon_range * 0.05
        e = east + lon_range * 0.05
        address = location.raw.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village") or city_name
        canton = address.get("state", "")
        display_name = f"{city}, {canton}, Switzerland"
        encoded = quote(display_name)
        return (
            f"https://flatfox.ch/en/search/?east={e}&north={n}&place_name={encoded}"
            f"&place_type=place&query={encoded}&south={s}&take=48&west={w}"
        )

    def start_requests(self):
        for city in self.SWISS_CITIES:
            url = self.get_city_url(city)
            if url:
                yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'city': city})

    def parse(self, response, city):
        self.driver.get(response.url)
        time.sleep(3)

        for _ in range(10): 
            try:
                WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Show more']"))
                ).click()
                time.sleep(2)
            except:
                break

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        cards = soup.select("div.listing-thumb")

        for card in cards:
            url_tag = card.select_one("a.listing-thumb__image")
            detail_url = response.urljoin(url_tag["href"]) if url_tag else None
            img_tag = card.select_one(".listing-image img")
            title = img_tag.get("alt") if img_tag else None
            price_tag = card.select_one("span.price")
            price_match = re.search(r"([\d'’]+)", price_tag.get_text() if price_tag else "")
            price = int(price_match.group(1).replace("'", "").replace("’", "")) if price_match else None
            address_text = card.select_one(".listing-thumb-title__location")
            zip_code, city_name = self.extract_location_fields(address_text.get_text() if address_text else "")

            item = {
                "title": title,
                "address": card.select_one("header.listing-thumb-header").get("title") if card.select_one("header.listing-thumb-header") else None,
                "zip_code": zip_code,
                "city": city_name or city,
                "region": city,
                "price": price,
            }

            if detail_url:
                yield scrapy.Request(detail_url, callback=self.parse_detail, cb_kwargs={'item': item}, dont_filter=True)

    def extract_location_fields(self, text):
        match = re.search(r'(\b\d{4})\s+([\wÀ-ÿ\'\- ]+)', text)
        return match.groups() if match else (None, None)

    def parse_detail(self, response, item):
        print(f"🧩 Parsing details for: {response.url}")
        self.driver.get(response.url)
        time.sleep(2)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        details = {}
        for tr in soup.select('tbody tr'):
            tds = tr.find_all('td')
            if len(tds) == 2:
                key = tds[0].get_text(strip=True)
                value = tds[1].get_text(strip=True)
                details[key] = value

        # Print extracted details
        print("✅ Extracted details:", details)

        item['rooms'] = details.get("Number of rooms:")
        item['floor'] = details.get("Floor:")
        item['area_sqm'] = details.get("Livingspace:")
        item['availability_date'] = details.get("Available:")

        facilities = details.get("Facilities:", "")
        item['has_balcony'] = "Balcony" in facilities

        desc_title = soup.select_one("strong.user-generated-content")
        desc_main = soup.select_one("div.markdown")

        desc_parts = []
        if desc_title:
            desc_parts.append(desc_title.get_text(strip=True))
        if desc_main:
            paragraphs = [p.get_text(strip=True) for p in desc_main.find_all("p")]
            desc_parts.extend(paragraphs)

        item['description'] = "\n\n".join(desc_parts) if desc_parts else None

        image_urls = []
        for fig in soup.select("figure[itemtype='http://schema.org/ImageObject'] a[itemprop='contentUrl']"):
            href = fig.get("href")
            if href:
                full_url = response.urljoin(href)
                image_urls.append(full_url)
        
        item['image_urls'] = image_urls
        yield item

    def closed(self, reason):
        self.driver.quit()
