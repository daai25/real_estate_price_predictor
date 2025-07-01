import scrapy


class PropertiesSpider(scrapy.Spider):
    name = "properties"
    allowed_domains = ["urbanhome.ch"]
    start_urls = ["https://urbanhome.ch"]

    def parse(self, response):
        pass
