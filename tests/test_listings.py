import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from house_price.listings import (
    ListingError,
    fetch_listing,
    parse_listing_text,
    parse_metadata,
    validate_url,
)

URL = "https://www.realtor.ca/real-estate/123456/example-toronto"


def metadata(**extra):
    record = {
        "@type": "House",
        "address": {"addressLocality": "Toronto", "addressRegion": "ON"},
        "numberOfBedrooms": "3 + 1",
        "numberOfBathroomsTotal": 2,
        "offers": {"price": 999999},
        **extra,
    }
    return '<script type="application/ld+json">' + json.dumps(record) + "</script>"


def test_parse_only_features_not_asking_price():
    assert parse_metadata(metadata()) == {"city": "Toronto", "bedrooms": 4, "bathrooms": 2}


def test_text_fallback():
    assert parse_listing_text("Toronto, Ontario · 3 + 1 Bedrooms · 2 Bathrooms · $999999") == {
        "city": "Toronto",
        "bedrooms": 4,
        "bathrooms": 2,
    }
    assert parse_listing_text("Bedrooms: 3 Bathrooms: 1.5 Hamilton") == {
        "city": "Hamilton",
        "bedrooms": 3,
        "bathrooms": 1.5,
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://www.realtor.ca/real-estate/123/a",
        "https://evil.example/real-estate/123/a",
        "https://realtor.ca.evil.example/real-estate/123/a",
        "https://127.0.0.1/real-estate/123/a",
        "https://user:pass@realtor.ca/real-estate/123/a",
        "https://realtor.ca:8080/real-estate/123/a",
        "https://realtor.ca/on/toronto/real-estate",
        "https://realtor.ca:invalid/real-estate/123/a",
    ],
)
def test_reject_unsafe_or_non_listing_urls(url):
    with pytest.raises(ListingError):
        validate_url(url)


def test_strip_tracking():
    assert validate_url(URL + "?tracking=123#anchor") == URL


def test_ambiguous_or_unavailable_metadata():
    for html in ["<html>Access denied</html>", metadata() + metadata()]:
        with pytest.raises(ListingError):
            parse_metadata(html)
    with pytest.raises(ListingError, match="outside Ontario"):
        parse_metadata(metadata(address={"addressLocality": "Vancouver", "addressRegion": "BC"}))


@pytest.mark.parametrize("status", [301, 403, 429, 500])
def test_http_failures_do_not_follow_redirects(status):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    with patch("house_price.listings.requests.get", return_value=response) as get:
        with pytest.raises(ListingError, match=f"HTTP {status}"):
            fetch_listing(URL)
        assert get.call_args.kwargs["allow_redirects"] is False


def test_timeout_fallback():
    with (
        patch("house_price.listings.requests.get", side_effect=requests.Timeout),
        pytest.raises(ListingError, match="Could not reach"),
    ):
        fetch_listing(URL)


def test_success_and_payload_limit():
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.headers = {"Content-Type": "text/html"}
    response.iter_content.return_value = [metadata().encode()]
    with patch("house_price.listings.requests.get", return_value=response):
        assert fetch_listing(URL)["bedrooms"] == 4
        response.iter_content.return_value = [b"x" * 2_000_001]
        with pytest.raises(ListingError, match="too large"):
            fetch_listing(URL)
