"""
hospital_finder.py
--------------------
FEATURE 2 — Nearby Hospitals.

Optional server-side helper that calls the Google Places "Nearby Search"
API to list real hospitals around a lat/lng. Only active if the
GOOGLE_PLACES_API_KEY environment variable is set.

If the key isn't set (or the request fails), find_nearby_hospitals()
returns an empty list — the frontend then falls back to a plain
"Open Nearby Hospitals in Google Maps" button, per the feature spec.
"""

import math
import requests
import config

NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


def is_configured():
    return config.is_places_configured()


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def find_nearby_hospitals(lat, lng, radius_m=5000, limit=6):
    """
    Returns a list of dicts: name, address, rating, distance_km, maps_url.
    Returns [] if the API key isn't configured or the request fails —
    callers should treat that as "use the fallback button" rather than an error.
    """
    api_key = config.get_places_api_key()
    if not api_key:
        return []

    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": "hospital",
        "key": api_key,
    }

    try:
        resp = requests.get(NEARBY_SEARCH_URL, params=params, timeout=6)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException:
        return []

    if payload.get("status") != "OK":
        return []

    hospitals = []
    for place in payload.get("results", [])[:limit]:
        loc = place.get("geometry", {}).get("location", {})
        plat, plng = loc.get("lat"), loc.get("lng")
        distance_km = round(_haversine_km(lat, lng, plat, plng), 1) if plat and plng else None
        place_id = place.get("place_id")
        maps_url = (
            f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            if place_id else
            f"https://www.google.com/maps/search/?api=1&query={plat},{plng}"
        )
        hospitals.append({
            "name": place.get("name", "Hospital"),
            "address": place.get("vicinity", ""),
            "rating": place.get("rating"),
            "distance_km": distance_km,
            "maps_url": maps_url,
        })

    hospitals.sort(key=lambda h: (h["distance_km"] is None, h["distance_km"]))
    return hospitals
