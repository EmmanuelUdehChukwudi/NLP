import scrapy
from scrapy import Field,Item

class BookscraperItem(scrapy.Item):
    title = Field()
    price = Field()
    upc = Field()
    img_url = Field()
    url = Field()
