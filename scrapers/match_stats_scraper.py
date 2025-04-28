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


def parse_map_header(header):
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


def parse_match_maps(soup: BeautifulSoup):
    header_divs = soup.find_all("div", class_="vm-stats-game-header")
    return [parse_map_header(h) for h in header_divs]


def get_team_name_and_tag(team_page: str) -> tuple[str, str]:
    resp = requests.get(team_page, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    container = soup.find("div", class_="team-header-name")

    team_name_tag = container.find("h1", class_="wf-title")
    team_name = team_name_tag.get_text(strip=True)

    team_tag_tag = container.find("h2", class_="wf-title team-header-tag")
    if team_tag_tag:
        team_tag = team_tag_tag.get_text(strip=True)
    else:
        team_tag = team_name

    return team_name, team_tag


def get_team_identifiers(soup: BeautifulSoup):
    team_a_page, team_b_page = extract_team_pages(soup)
    team_a_name, team_a_tag = get_team_name_and_tag(team_a_page)
    team_b_name, team_b_tag = get_team_name_and_tag(team_b_page)
    team_a = {"name": team_a_name, "tag": team_a_tag, "page": team_a_page}
    team_b = {"name": team_b_name, "tag": team_b_tag, "page": team_b_page}
    return team_a, team_b


def parse_match(soup: BeautifulSoup):
    team_a, team_b = get_team_identifiers(soup)
    match_data = {
        "match_url": match_url,
        "match_date": parse_match_date(soup),
        "team_a": team_a,
        "team_b": team_b,
        "player_stats": get_player_stats(soup),
        "maps_stats": parse_match_maps(soup),
    }
    return match_data


def get_team_match_page(team_page: str):
    pattern = r"^(https://www\.vlr\.gg)/team/(\d+)/(.*?)/?$"
    repl = r"\1/team/matches/\2/\3/?group=completed"
    return re.sub(pattern, repl, team_page)


def get_prev_n_match_urls(
    original_match_url: str, team_page: str, excluded_team: str, n: int = 7
):
    print(team_page)
    # NOTE: this will only work for matches that appear on the 1st page of the match history
    team_match_page = get_team_match_page(team_page)

    resp = requests.get(team_match_page, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    container = soup.find("div", class_="mod-dark")
    rows = container.select("a.wf-card.m-item")

    matches = []
    for a in rows:
        new_match_url = urljoin(BASE_URL, a["href"].split("?")[0])
        names = set([s.get_text(strip=True) for s in a.select("span.m-item-team-name")])
        if new_match_url == original_match_url or (
            excluded_team not in names
            and any([t.lower() in new_match_url for t in REGIONAL_TOURNAMENTS])
        ):
            matches.append(new_match_url)

    idx = matches.index(original_match_url)
    return matches[idx + 1 : idx + 1 + n]


def get_prev_n_h2h_match_urls(soup: BeautifulSoup, n: int = 3):
    urls = []
    container = soup.find("div", class_="match-h2h-matches")
    if not container:
        return urls

    for a in container.find_all("a", href=True):
        full = urljoin(BASE_URL, a["href"])
        urls.append(full)

    return urls[:n]


def aggregate_prev_match_map_stats(match_urls: list[str], team_name: str):
    total_map_wins = total_maps = total_round_wins = total_rounds = 0
    for match_url in match_urls:
        resp = requests.get(match_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        match_data = parse_match(soup)

        total_maps += len(match_data["maps_stats"])
        total_map_wins += sum(
            [
                int(
                    m["team_a_score"] > m["team_b_score"]
                    if m["team_a_name"] == team_name
                    else m["team_b_score"] > m["team_a_score"]
                )
                for m in match_data["maps_stats"]
            ]
        )
        total_round_wins += sum(
            [
                (
                    m["team_a_score"]
                    if m["team_a_name"] == team_name
                    else m["team_b_score"]
                )
                for m in match_data["maps_stats"]
            ]
        )
        total_rounds += sum(
            [(m["team_a_score"] + m["team_b_score"]) for m in match_data["maps_stats"]]
        )

    df = pd.DataFrame(
        [
            {
                "Round Win Pct": total_round_wins / total_rounds,
                "Map Win Pct": total_map_wins / total_maps,
            }
        ]
    )
    return df


def aggregate_prev_matches_player_stats(
    original_match_url: str, match_urls: list[str], team_tag: str, team_name: str
) -> pd.DataFrame:
    player_stats = []
    for match_url in match_urls:
        resp = requests.get(match_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        match_data = parse_match(soup)
        total_rounds = sum(
            m["team_a_score"] + m["team_b_score"] for m in match_data["maps_stats"]
        )

        df = match_data["player_stats"]
        team_row = df[df["team_tag"] == team_tag]
        team_row = team_row.assign(
            **{
                "Rating": lambda d: d["Rating 2.0"],
                "Kills:Deaths": lambda d: d["Kills"] / d["Deaths"],
                "First Kills Per Round": lambda d: d["First Kills"] / total_rounds,
                "First Deaths Per Round": lambda d: d["First Deaths"] / total_rounds,
                "Kill, Assist, Trade, Survive %": lambda d: d[
                    "Kill, Assist, Trade, Survive %"
                ]
                / 100,
            }
        )

        player_stats.append(team_row)

    all_stats = pd.concat(player_stats, ignore_index=True)
    team_stats = all_stats[all_stats["team_tag"] == team_tag]
    agg_stats = team_stats.drop(columns="team_tag").mean()
    df = (
        agg_stats.loc[
            [
                "Rating",
                "Average Combat Score",
                "Kills:Deaths",
                "Kill, Assist, Trade, Survive %",
                "First Kills Per Round",
                "First Deaths Per Round",
            ]
        ]
        .to_frame()
        .T
    )
    df.insert(0, "Matchup", original_match_url)
    df.insert(1, "Team Name", team_name)
    return df


# store the transformed data in a csv
def create_teams_matchups_stats_df(
    match_url,
    team_a_vs_b_stats,
    team_b_vs_a_stats,
    team_a_vs_others_stats,
    team_b_vs_others_stats,
):
    return pd.DataFrame(
        {
            "Matchup": [match_url] * 4,
            "Team": ["A", "A", "B", "B"],
            "Opponent": ["B", "Others", "A", "Others"],
            "Round Win Pct": [
                team_a_vs_b_stats["Round Win Pct"].values[0],
                team_a_vs_others_stats["Round Win Pct"].values[0],
                team_b_vs_a_stats["Round Win Pct"].values[0],
                team_b_vs_others_stats["Round Win Pct"].values[0],
            ],
            "Map Win Pct": [
                team_a_vs_b_stats["Map Win Pct"].values[0],
                team_a_vs_others_stats["Map Win Pct"].values[0],
                team_b_vs_a_stats["Map Win Pct"].values[0],
                team_b_vs_others_stats["Map Win Pct"].values[0],
            ],
        }
    )


def create_teams_matchups_stats_csv(all_matchup_stats: dict):
    combined_df = pd.concat(all_matchup_stats.values()).reset_index()
    repeat_count = len(combined_df) // len(all_matchup_stats)

    matchup_identifiers = []
    for key in all_matchup_stats.keys():
        matchup_identifiers.extend([key] * repeat_count)

    combined_df["Matchup"] = matchup_identifiers
    print(combined_df)


def create_players_stats_csv(all_players_stats: list):
    print(all_players_stats)


if __name__ == "__main__":
    set_display_options()
    all_matchup_stats = {}
    all_players_stats = []
    for match_url in list(load_year_match_odds_from_csv("2024").keys()):
        resp = requests.get(match_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        print(f"\n{match_url}")
        current_match_data = parse_match(soup)
        team_a_name = current_match_data["team_a"]["name"]
        team_b_name = current_match_data["team_b"]["name"]

        team_a_prev_matches_urls = get_prev_n_match_urls(
            original_match_url=match_url,
            team_page=current_match_data["team_a"]["page"],
            excluded_team=team_b_name,
        )
        team_b_prev_matches_urls = get_prev_n_match_urls(
            original_match_url=match_url,
            team_page=current_match_data["team_b"]["page"],
            excluded_team=team_a_name,
        )
        prev_h2h_urls = get_prev_n_h2h_match_urls(soup, n=3)

        team_a_agg_player_stats = aggregate_prev_matches_player_stats(
            original_match_url=match_url,
            match_urls=team_a_prev_matches_urls,
            team_tag=current_match_data["team_a"]["tag"],
            team_name=team_a_name,
        )
        team_a_agg_map_stats = aggregate_prev_match_map_stats(
            match_urls=team_a_prev_matches_urls, team_name=team_a_name
        )
        team_b_agg_player_stats = aggregate_prev_matches_player_stats(
            original_match_url=match_url,
            match_urls=team_b_prev_matches_urls,
            team_tag=current_match_data["team_b"]["tag"],
            team_name=team_b_name,
        )
        team_b_agg_map_stats = aggregate_prev_match_map_stats(
            match_urls=team_b_prev_matches_urls, team_name=team_b_name
        )
        team_a_h2h_agg_map_stats = aggregate_prev_match_map_stats(
            match_urls=prev_h2h_urls, team_name=team_a_name
        )
        team_b_h2h_agg_map_stats = aggregate_prev_match_map_stats(
            match_urls=prev_h2h_urls, team_name=team_b_name
        )

        matchup_stats_df = create_teams_matchups_stats_df(
            match_url,
            team_a_h2h_agg_map_stats,
            team_b_h2h_agg_map_stats,
            team_a_agg_map_stats,
            team_b_agg_map_stats,
        )

        all_matchup_stats[match_url] = matchup_stats_df
        all_players_stats.append(team_a_agg_player_stats)
        all_players_stats.append(team_b_agg_player_stats)

    all_players_stats_df = pd.concat(all_players_stats, axis=0)

    combined_matchup_stats_df = pd.concat(all_matchup_stats.values()).reset_index()
    repeat_count = len(combined_matchup_stats_df) // len(all_matchup_stats)

    matchup_identifiers = []
    for key in all_matchup_stats.keys():
        matchup_identifiers.extend([key] * repeat_count)

    combined_matchup_stats_df["Matchup"] = matchup_identifiers

    combined_matchup_stats_df.to_csv(
        "scraped_data/matches/teams_matchups_stats.csv", index=False
    )
    combined_matchup_stats_df.to_csv(
        "scraped_data/players_stats/team_players_stats.csv", index=False
    )
