import scrapy


class PropertiesSpider(scrapy.Spider):
    name = "properties"
    allowed_domains = ["flatfox.ch"]
    start_urls = ["https://flatfox.ch"]

    def parse(self, response):
        pass
