"""Best-effort public Realtor.ca metadata import; never use the asking price."""

import json
import re
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from house_price.data import CITIES


class ListingError(ValueError):
    """An actionable import failure that does not block manual estimation."""


def validate_url(url):
    try:
        parsed = urlsplit(url.strip())
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in {"realtor.ca", "www.realtor.ca"}
            and parsed.port in {None, 443}
            and not parsed.username
            and not parsed.password
            and re.fullmatch(r"/real-estate/\d+/[A-Za-z0-9%-]+/?", parsed.path)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ListingError("Paste an HTTPS Realtor.ca property link containing /real-estate/.")
    return urlunsplit(("https", parsed.hostname, parsed.path, "", ""))


def _city(value):
    value = str(value).strip()
    for city in sorted(CITIES, key=len, reverse=True):
        if re.fullmatch(re.escape(city) + r"(?:\s*\([^)]*\))?", value, flags=re.IGNORECASE):
            return city
    return None


def _number(value):
    if isinstance(value, dict):
        value = value.get("value")
    match = re.fullmatch(r"\s*(\d+(?:\.5)?)(?:\s*\+\s*(\d+))?\s*", str(value))
    if not match:
        return None
    result = float(match[1]) + float(match[2] or 0)
    return result if 0 <= result <= 20 else None


def parse_metadata(html):
    """Use explicit residence JSON-LD only, avoiding unrelated cards and price fields."""
    soup = BeautifulSoup(html, "html.parser")
    residences = []

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            kind = value.get("@type", [])
            kinds = [kind] if isinstance(kind, str) else kind
            if not isinstance(kinds, list):
                kinds = []
            kinds = [item for item in kinds if isinstance(item, str)]
            if set(kinds) & {"House", "SingleFamilyResidence", "Apartment", "Residence"}:
                residences.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            walk(json.loads(script.string or script.get_text()))
        except (ValueError, TypeError):
            continue
    if len(residences) != 1:
        raise ListingError(
            "No single unambiguous home record was available. Paste listing details below."
        )
    record = residences[0]
    address = record.get("address", {})
    if not isinstance(address, dict):
        address = {}
    region = str(address.get("addressRegion", "")).casefold()
    if region and region not in {"on", "ontario"}:
        raise ListingError("This listing is outside Ontario, the model's supported area.")
    result = {
        "city": _city(address.get("addressLocality", "")),
        "bedrooms": _number(record.get("numberOfBedrooms")),
        "bathrooms": _number(record.get("numberOfBathroomsTotal")),
    }
    return {key: value for key, value in result.items() if value is not None}


def parse_listing_text(text):
    """Extract explicit room counts and a supported city from user-pasted listing text."""
    result = {}
    for key, pattern in [("bedrooms", r"bedrooms?|beds?"), ("bathrooms", r"bathrooms?|baths?")]:
        number = r"(\d+(?:\.5)?(?:\s*\+\s*\d+)?)"
        # Explicit label:value wins over a preceding count belonging to another label.
        match = re.search(r"\b(?:" + pattern + r")\s*:\s*" + number, text, re.IGNORECASE)
        if not match:
            match = re.search(number + r"\s*(?:" + pattern + r")\b", text, re.IGNORECASE)
        if not match:
            match = re.search(r"\b(?:" + pattern + r")\s+" + number, text, re.IGNORECASE)
        if match and (value := _number(match[1])) is not None:
            result[key] = value
    found = [
        city for city in CITIES if re.search(r"\b" + re.escape(city) + r"\b", text, re.IGNORECASE)
    ]
    if len(found) == 1:
        result["city"] = found[0]
    return result


def fetch_listing(url):
    url = validate_url(url)
    try:
        # Fixed hosts, no redirects, bounded payload, no login/cookie/anti-bot workarounds.
        with requests.get(
            url,
            timeout=(5, 10),
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "OntarioPricePortfolio/0.1 (public metadata reader)"},
        ) as response:
            if response.status_code != 200:
                raise ListingError(
                    f"Realtor.ca did not provide the page (HTTP {response.status_code}). "
                    "Paste the listing text or enter the details manually."
                )
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise ListingError(
                    "The link did not return a listing page. Enter details manually."
                )
            chunks, size = [], 0
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > 2_000_000:
                    raise ListingError("Listing page is too large. Paste its details instead.")
                chunks.append(chunk)
            result = parse_metadata(b"".join(chunks).decode("utf-8", errors="replace"))
    except requests.RequestException as exc:
        raise ListingError(
            "Could not reach Realtor.ca. Paste listing text or enter details manually."
        ) from exc
    if not result:
        raise ListingError("No supported housing features found. Enter the details manually.")
    return result
