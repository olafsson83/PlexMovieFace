"""Thin Plex API wrapper, used only by setup.py's optional "look up movie via
Plex" helper to resolve a title pick into a local file path. The actual
discovery/swap pipeline never talks to Plex once MOVIE_PATH is set.
"""
import requests

from config import PLEX_URL, PLEX_TOKEN

_JSON_HEADERS = {"X-Plex-Token": PLEX_TOKEN, "Accept": "application/json"}


def get_sections():
    """Returns the server's library sections, e.g. [{"key": "1", "title": "Movies", ...}, ...]."""
    r = requests.get(f"{PLEX_URL}/library/sections", headers=_JSON_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["MediaContainer"].get("Directory", [])


def get_section_items(section_key):
    """Returns the top-level items (movies) in a library section."""
    r = requests.get(f"{PLEX_URL}/library/sections/{section_key}/all", headers=_JSON_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["MediaContainer"].get("Metadata", [])


def get_movie_file_path(rating_key: str):
    """Returns the local filesystem path Plex has on disk for a movie item,
    or None if it can't be determined (e.g. the item has no local file, or
    PlexMovieFace isn't running on a machine with access to Plex's filesystem).
    """
    r = requests.get(f"{PLEX_URL}/library/metadata/{rating_key}", headers=_JSON_HEADERS, timeout=30)
    r.raise_for_status()
    items = r.json()["MediaContainer"].get("Metadata", [])
    if not items:
        return None
    for media in items[0].get("Media", []):
        for part in media.get("Part", []):
            if part.get("file"):
                return part["file"]
    return None
