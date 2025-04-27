import re, requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

import pandas as pd

from scrapers import HEADERS, BASE_URL
from helper import (
    load_year_match_odds_from_csv,
    set_display_options,
    REGIONAL_TOURNAMENTS,
)


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
    vs_block = soup.find("div", class_="match-header-vs")

    team_links = []
    for a in vs_block.find_all("a", class_="match-header-link"):
        href = a.get("href", "")
        full = urljoin(BASE_URL, href)
        if full not in team_links:
            team_links.append(full)

    return team_links


def parse_table(table, headers):
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")

        player_a = cells[0].find("a")
        player_url = urljoin(BASE_URL, player_a["href"])
        player_name = player_a.find("div", class_="text-of").get_text(strip=True)
        team_tag = player_a.find("div", class_="ge-text-light").get_text(strip=True)
        data = {"team_tag": team_tag, "player": player_name, "profile_url": player_url}

        for header, cell in zip(headers[2:], cells[2:]):
            val = cell.select_one("span.side.mod-both").get_text(strip=True)
            data[header] = val

        rows.append(data)

    return rows


def get_player_stats(soup: BeautifulSoup):
    container = soup.find(
        "div", class_="vm-stats-game mod-active", attrs={"data-game-id": "all"}
    )

    team_stats = []
    for team_table in container.find_all("table", class_="wf-table-inset mod-overview"):
        table_headers = [
            th.get("title") or th.get_text(strip=True)
            for th in team_table.find("thead").find_all("th")
        ]
        player_data = parse_table(team_table, table_headers)
        df = pd.DataFrame(player_data)

        stat_cols = [
            c for c in df.columns if c not in ("team_tag", "player", "profile_url")
        ]
        for c in stat_cols:
            df[c] = df[c].astype(str).str.rstrip("%").str.lstrip("+").astype(float)

        team_avg = {"team_tag": df["team_tag"].iloc[0]}
        for c in stat_cols:
            team_avg[c] = df[c].mean()

        team_stats.append(team_avg)

    return pd.DataFrame(team_stats)


def parse_map_header(header: BeautifulSoup):
    team_a = header.find("div", class_="team")
    team_a_name = team_a.find("div", class_="team-name").get_text(strip=True)
    team_a_score = int(team_a.find("div", class_="score").get_text(strip=True))

    team_b = header.find("div", class_="team mod-right")
    team_b_name = team_b.find("div", class_="team-name").get_text(strip=True)
    team_b_score = int(team_b.find("div", class_="score").get_text(strip=True))

    map_div = header.find("div", class_="map")
    name_div = map_div.find("div", style=lambda s: s and "font-weight" in s)

    pick_span = name_div.find("span", class_="picked")
    if pick_span:
        classes = pick_span.get("class", [])
        pick_cls = next((c for c in classes if c.startswith("mod-")), None)
        picked_by = team_a_name if pick_cls == "mod-1" else team_b_name
    else:
        picked_by = "DECIDER"

    outer_span = name_div.find("span", recursive=False)
    map_name = outer_span.contents[0].strip()

    return {
        "map_name": map_name,
        "team_a_name": team_a_name,
        "team_a_score": team_a_score,
        "team_b_name": team_b_name,
        "team_b_score": team_b_score,
        "picked_by": picked_by,
    }


def parse_match_maps(soup):
    header_divs = soup.find_all("div", class_="vm-stats-game-header")
    return [parse_map_header(h) for h in header_divs]


def parse_match(match_url: str):
    resp = requests.get(match_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    team_a_page, team_b_page = extract_team_pages(soup)
    match_data = {
        "match_url": match_url,
        "match_date": parse_match_date(soup),
        "team_a_page": team_a_page,
        "team_b_page": team_b_page,
        "player_stats": get_player_stats(soup),
        "maps_data": parse_match_maps(soup),
    }
    return match_data


def get_team_match_page(team_page: str):
    pattern = r"^(https://www\.vlr\.gg)/team/(\d+)/(.*?)/?$"
    repl = r"\1/team/matches/\2/\3/?group=completed"
    return re.sub(pattern, repl, team_page)


def get_prev_n_match_urls(original_match_url: str, team_page: str, n: int = 7):
    # NOTE: this will only work for matches that appear on the 1st page of the match history
    team_match_page = get_team_match_page(team_page)

    resp = requests.get(team_match_page, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    container = soup.find("div", class_="mod-dark")
    rows = container.select("a.wf-card.m-item")

    matches, seen = [], set()
    for a in rows:
        base = urljoin(BASE_URL, a["href"].split("?")[0])
        if base not in seen and any([t.lower() in base for t in REGIONAL_TOURNAMENTS]):
            seen.add(base)
            matches.append(base)

    idx = matches.index(original_match_url)
    return matches[idx + 1 : idx + 1 + n]


if __name__ == "__main__":
    set_display_options()

    for match_url in list(load_year_match_odds_from_csv("2024").keys())[:4]:
        print(f"\n{match_url}")
        current_match_data = parse_match(match_url)
        print(current_match_data["maps_data"])
        team_a_prev_matches = get_prev_n_match_urls(
            match_url, current_match_data["team_a_page"]
        )
        team_b_prev_matches = get_prev_n_match_urls(
            match_url, current_match_data["team_b_page"]
        )

        # for match_url in team_a_prev_matches:
        #     match_data = parse_match(match_url)

        # for match_url in team_b_prev_matches:
        #     match_data = parse_match(match_url)
