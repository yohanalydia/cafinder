"""
Google Maps / Places API integration.

Tujuan:
Ambil detail cafe dari Google Maps
berdasarkan nama cafe hasil rekomendasi.

Contoh:
Layung Coffee Roastery
↓
Google Places API
↓
alamat, koordinat, rating maps
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    ""
)


def search_place(
    place_name: str,
    lokasi: str = ""
):
    """
    Cari cafe di Google Maps.

    Input:
        place_name:
            Layung Coffee Roastery

        lokasi:
            Bandung Timur

    Return:
        dictionary detail cafe
    """

    if not GOOGLE_MAPS_API_KEY:
        return None

    query = f"{place_name} {lokasi}"

    url = (
        "https://maps.googleapis.com/"
        "maps/api/place/textsearch/json"
    )

    params = {
        "query": query,
        "key": GOOGLE_MAPS_API_KEY,
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    if not data.get("results"):
        return None

    place = data["results"][0]

    return {
        "name":
            place.get("name"),

        "address":
            place.get(
                "formatted_address"
            ),

        "rating":
            place.get("rating"),

        "lat":
            place.get(
                "geometry",
                {}
            )
            .get(
                "location",
                {}
            )
            .get("lat"),

        "lng":
            place.get(
                "geometry",
                {}
            )
            .get(
                "location",
                {}
            )
            .get("lng"),

        "place_id":
            place.get(
                "place_id"
            ),
    }


def build_maps_link(
    lat,
    lng
):
    """
    Buat link Google Maps.

    User klik
    ↓
    buka maps
    """

    if lat is None or lng is None:
        return None

    return (
        "https://www.google.com/maps"
        f"?q={lat},{lng}"
    )

def get_map_df(
    lat,
    lng
):

    if lat is None or lng is None:
        return None

    import pandas as pd

    return pd.DataFrame(
        {
            "lat":[lat],
            "lon":[lng],
        }
    )

if __name__ == "__main__":

    result = search_place(
        "Layung Coffee Roastery",
        "Bandung Timur"
    )

    print(result)

    if result:

        print(
            build_maps_link(
                result["lat"],
                result["lng"]
            )
        )