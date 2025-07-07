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
        "Fribourg", "Chur", "Neuchâtel", "Vernier", "Uster", "Sion",
        "Yverdon-les-Bains", "Zug", "Rapperswil-Jona", "Dietikon", "Montreux", "Frauenfeld", "Wil",
        "Baar", "Bellinzona", "Carouge", "Locarno", "Meyrin", "Wädenswil", "Wetzikon", "Bulle",
        "Aarau", "Gossau", "Muttenz", "Kreuzlingen", "Allschwil", "Olten", "Pully", "Burgdorf",
        "Vevey", "Martigny", "Renens", "Emmen", "Sierre", "Hinwil", "Thalwil", "Romanshorn",
        "Baden", "Lancy", "Pfäffikon", "Arbon", "Solothurn", "Steffisburg", "Neuenhof",
        "Glarus", "Chiasso", "Schwyz", "Liestal", "Brig", "Herisau"
    ]

    def __init__(self):
        """
        Initializes the web driver for Chrome browser.

        This constructor sets up the Chrome web driver using the ChromeDriverManager to
        automatically manage the driver binary. Optionally, specific Chrome options
        can be configured, such as enabling headless mode.

        Attributes:
            driver: Represents the initialized Selenium WebDriver instance for the
                Chrome browser.
        """
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def get_city_url(self, city_name):
        """
        Generate a URL for a real estate search in Flatfox based on the given city name.

        This method uses the geopy library's Nominatim geocoder to retrieve geographical
        coordinates and location details for the specified city name. It then calculates
        an expanded bounding box around the city's location to include slightly larger
        regions in the search query. A URL is generated for use in the Flatfox search
        platform, targeting the specified location and its vicinity.

        Parameters:
            city_name (str): The name of the city to generate the Flatfox search URL.

        Returns:
            str or None: A complete URL for a Flatfox search localized to the city, or
            None if the city cannot be resolved by the geocoder.

        Raises:
            None explicitly. Exceptions from the geopy library may occur during the
            geocode process if the geocoder API cannot be reached or does not find the location.
        """
        # Initialize the geocoder with a user agent
        geolocator = Nominatim(user_agent="flatfox_scraper")
        location = geolocator.geocode(city_name, exactly_one=True, addressdetails=True, bounded=True)
        if not location:
            return None
        # Calculate an expanded bounding box around the city location
        south, north, west, east = map(float, location.raw["boundingbox"])
        lat_range = north - south
        lon_range = east - west
        # Expand the bounding box by 5% in each direction
        s = south - lat_range * 0.05
        n = north + lat_range * 0.05
        w = west - lon_range * 0.05
        e = east + lon_range * 0.05
        # Construct the Flatfox search URL
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
        """
        Initiates HTTP requests for a list of Swiss cities.

        This method generates a request for each city in the `SWISS_CITIES` list. The
        `get_city_url` method is used to construct the specific URL for a city. If a valid
        URL is returned, a Scrapy Request object is yielded with the URL, parse callback,
        and the city name passed as a keyword argument.

        Yields:
            scrapy.Request: A Scrapy Request configured with the city-specific URL, parse
            callback, and city context in `cb_kwargs`.
        """
        for city in self.SWISS_CITIES:
            # Generate the URL for the city
            url = self.get_city_url(city)
            if url:
                yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'city': city})

    def parse(self, response, city):
        """
        Parses the response and scrapes property listing data, including title, address, zip code,
        city, region, price, and detail URLs. It interacts with a dynamic webpage using Selenium
        for loading additional content and utilizes BeautifulSoup for parsing the HTML content.

        Parameters:
            response (scrapy.http.Response): The response object containing the URL and content
                to parse.
            city (str): The name of the city associated with the listing.

        Yields:
            scrapy.Request: A request object to fetch detailed information about a specific listing.
        """
        # Navigate to the response URL using Selenium
        self.driver.get(response.url)
        time.sleep(3)

        # Click on "Show more" button to load additional listings
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

        # Extract property listings from the soup object
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
        """
        Extracts location information from a given text.

        This function searches for a specific pattern within the input text that matches
        a location format consisting of a 4-digit number followed by a location name. If
        a match is found, it returns the corresponding groups extracted from the pattern.
        If no match is found, it returns a tuple of (None, None).

        Parameters:
        text: str
            The input text in which to search for location information.

        Returns:
        tuple
            A tuple containing two elements. The first element represents the 4-digit
            location identifier, and the second element represents the name of the
            location. Returns (None, None) if no match is found.
        """
        match = re.match(r'(\d{4})\s+(.*)', text.strip())
        return match.groups() if match else (None, None)

    def parse_detail(self, response, item):
        """
        Parses the details page of a property listing and extracts relevant information
        such as room count, area, availability date, description, facilities, and image
        URLs. The extracted information is stored in the provided `item` dictionary.

        Parameters:
        response (Response): The response object representing the details page to be parsed.
        item (dict): A dictionary to store the extracted property details.

        Returns:
        Generator: Yields the updated `item` dictionary populated with extracted details.

        Raises:
        None
        """
        # Log the URL being parsed
        print(f"Parsing details for: {response.url}")
        self.driver.get(response.url)
        time.sleep(2)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # Extract details from the page
        details = {}
        for tr in soup.select('tbody tr'):
            tds = tr.find_all('td')
            if len(tds) == 2:
                key = tds[0].get_text(strip=True)
                value = tds[1].get_text(strip=True)
                details[key] = value

        # Print extracted details
        print("Extracted details:", details)

        item['rooms'] = details.get("Number of rooms:")
        item['floor'] = details.get("Floor:")
        item['area_sqm'] = details.get("Livingspace:")
        item['availability_date'] = details.get("Available:")

        facilities = details.get("Facilities:", "")
        item['has_balcony'] = "Balcony" in facilities

        desc_title = soup.select_one("strong.user-generated-content")
        desc_main = soup.select_one("div.markdown")

        # Extract description parts
        desc_parts = []
        if desc_title:
            desc_parts.append(desc_title.get_text(strip=True))
        if desc_main:
            paragraphs = [p.get_text(strip=True) for p in desc_main.find_all("p")]
            desc_parts.extend(paragraphs)

        item['description'] = "\n\n".join(desc_parts) if desc_parts else None

        # Extract image URLs
        image_urls = []
        for fig in soup.select("figure[itemtype='http://schema.org/ImageObject'] a[itemprop='contentUrl']"):
            href = fig.get("href")
            if href:
                full_url = response.urljoin(href)
                image_urls.append(full_url)
        
        item['image_urls'] = image_urls
        yield item

    def closed(self, reason):
        """
        Closes the current driver session.

        This method shuts down the driver and ends the current session by quitting the
        driver instance. It is typically used to properly close resources and terminate
        interactions with the driver.

        Args:
            reason (str): The explanation or context for closing the driver session.
        """
        self.driver.quit()
