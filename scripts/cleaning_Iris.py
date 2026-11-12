# translated parsing code from R to Python
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def clean_html(html):
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = text.replace("'", "").replace("nbsp", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()