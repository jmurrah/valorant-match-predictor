import csv
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from scrapers import HEADERS, BASE_URL
from helper import GLOBAL_TOURNAMENTS, REGIONAL_TOURNAMENTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_tournament_match_urls(tournament_url: str) -> list[str]:
    pattern = re.compile(r"^/\d+/.+")
    html = requests.get(tournament_url, headers=HEADERS, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    links = {urljoin(BASE_URL, a["href"]) for a in soup.find_all("a", href=pattern)}
    time.sleep(1)
    return sorted(links)


def get_yearly_match_urls(year: str) -> list[str]:
    """Return all regional-kickoff match URLs for the given VCT year."""
    year_page = requests.get(f"{BASE_URL}/vct-{year}", headers=HEADERS, timeout=20)
    soup = BeautifulSoup(year_page.content, "html.parser")

    tournament_cards = (
        soup.find("div", class_="events-container")
        .find_all("div", class_="events-container-col")[-1]
        .find_all("a", class_="wf-card mod-flex event-item")
    )

    yearly_match_urls: list[str] = []
    for card in tournament_cards:
        href = card.get("href")
        tournament_matches_url = BASE_URL + href.replace("/event/", "/event/matches/")
        tournament = (
            card.find("div", class_="event-item-title").text.strip().split(": ")
        )

        # only regional tournaments for now
        if len(tournament) == 2 and any(
            t in tournament[-1] for t in REGIONAL_TOURNAMENTS
        ):
            yearly_match_urls.extend(get_tournament_match_urls(tournament_matches_url))

    return yearly_match_urls


def fetch_with_js(url: str) -> BeautifulSoup | None:
    """
    Load the page in Playwright; return BeautifulSoup tree
    or None if Thunderpick never shows or any error occurs.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])

            page.goto(url, timeout=25_000, wait_until="domcontentloaded")

            for attempt in range(1, 4):
                try:
                    page.wait_for_selector(
                        "img[src*='thunderpick']", state="attached", timeout=5_000
                    )
                    break
                except PWTimeout:
                    if attempt < 3:
                        log.info("Thunderpick timeout → refreshing page")
                        page.reload(wait_until="domcontentloaded")
                    else:
                        log.warning("Thunderpick still missing → skip match")
                        return None

            html = page.content()
            browser.close()
            return BeautifulSoup(html, "html.parser")

    except Exception as e:
        log.warning(f"Playwright error on {url}: {e}")
        return None


def get_match_odds(url: str) -> dict[str, float]:
    """
    Return {team: odd, team: odd} for a single VLR match,
    or {} if odds cannot be parsed.
    """
    soup = fetch_with_js(url)
    if soup is None:
        return {}

    try:
        team_a_elem = soup.select_one(".match-header-link-name.mod-1 .wf-title-med")
        team_b_elem = soup.select_one(".match-header-link-name.mod-2 .wf-title-med")
        odds_img = soup.select_one("img[src*='thunderpick']")

        if not (team_a_elem and team_b_elem and odds_img):
            raise ValueError("Required elements missing")

        team_a = team_a_elem.get_text(strip=True)
        team_b = team_b_elem.get_text(strip=True)

        odds_div = odds_img.find_parent("div").find_parent("div")
        text = odds_div.get_text(" ", strip=True)

        winning_team = re.search(r"on\s+(.+?)\s+returned", text).group(1)

        d1_raw = float(
            re.search(r"at\s+([0-9]+(?:\.[0-9]+)?)\s+pre-match odds", text).group(1)
        )

        q1 = 1.0 / d1_raw
        R = 1.0 + 0.08  # remove the vig of 8%
        q2 = R - q1

        q1_fair = q1 / R
        q2_fair = q2 / R

        d1_fair = 1.0 / q1_fair
        d2_fair = 1.0 / q2_fair

        return {
            team_a: round(d1_fair, 4) if team_a == winning_team else round(d2_fair, 4),
            team_b: round(d1_fair, 4) if team_b == winning_team else round(d2_fair, 4),
        }

    except Exception as e:
        log.warning(f"Parsing error on {url}: {e}")
        return {}


def save_year_match_odds_to_csv(
    year: str, match_odds: dict[str, dict[str, float]]
) -> None:
    """Write one CSV row per match: match_url, team_a, odd_a, team_b, odd_b."""
    out_file = (
        Path("scraped_data/thunderpick_match_odds")
        / f"{year}_thunderpick_match_odds.csv"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["match_url", "team_a", "odd_a", "team_b", "odd_b"])
        for match_url, teams in match_odds.items():
            (team_a, odd_a), (team_b, odd_b) = teams.items()
            writer.writerow([match_url, team_a, odd_a, team_b, odd_b])


if __name__ == "__main__":
    years = ["2024"]

    for year in years:
        yearly_match_odds = defaultdict(dict)

        for match_url in get_yearly_match_urls(year):
            if "americas" not in match_url:
                continue
            log.info(f"→ {match_url}")
            odds = get_match_odds(match_url)
            if not odds:
                continue

            yearly_match_odds[match_url] = odds
            for team, odd in odds.items():
                log.info(f"   {team}: {odd:.2f}")

        save_year_match_odds_to_csv(year, yearly_match_odds)
        log.info(f"Saved {len(yearly_match_odds)} matches for {year}")
