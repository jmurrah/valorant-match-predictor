import re, os, requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

from scrapers import HEADERS
from helper import load_year_match_odds_from_csv


def parse_match_date(soup: BeautifulSoup):
    date_container = soup.find("div", class_="match-header-date")
    moments = date_container.find_all("div", class_="moment-tz-convert")

    utc_ts = moments[0]["data-utc-ts"]
    match_dt = datetime.fromisoformat(utc_ts)

    date_str = moments[0].get_text(strip=True)
    time_str = moments[1].get_text(strip=True)

    return {
        "datetime_utc": match_dt,
        "date_str": date_str,
        "time_str": time_str,
    }


def extract_team_pages(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Given a BeautifulSoup for a match page, return the two absolute URLs
    for the teams that are playing.
    """
    base = "https://www.vlr.gg"
    vs_block = soup.find("div", class_="match-header-vs")

    team_links = []
    for a in vs_block.find_all("a", class_="match-header-link"):
        href = a.get("href", "")
        full = urljoin(base, href)
        if full not in team_links:
            team_links.append(full)

    return tuple(team_links)


def parse_match(match_url: str):
    resp = requests.get(match_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    match_date = parse_match_date(soup)
    team_pages = extract_team_pages(soup)
    print(match_date)
    print(team_pages)


if __name__ == "__main__":
    print("hello")
    for match_url in list(load_year_match_odds_from_csv("2024").keys())[:3]:
        print(f"\n{match_url}")
        parse_match(match_url)
