from .alfworld import AlfWorldExtractor
from .webshop import WebShopExtractor


def get_extractor(env_name: str):
    if "webshop" in env_name.lower():
        return WebShopExtractor()
    return AlfWorldExtractor()
