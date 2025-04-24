import re
import requests
import time
from bs4 import BeautifulSoup


def get_match_odds(url: str) -> dict[str, float]:
    """
    Fetches the given VLR match URL and returns a dict mapping
    each team name to its pre-match odds from Thunderpick.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        "AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/114.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    time.sleep(1)

    soup = BeautifulSoup(resp.text, "html.parser")
    team_a_elem = soup.select_one(".match-header-link-name.mod-1 .wf-title-med")
    team_b_elem = soup.select_one(".match-header-link-name.mod-2 .wf-title-med")

    team_a = team_a_elem.get_text(strip=True)
    team_b = team_b_elem.get_text(strip=True)

    odds_div = None
    for img in soup.select("img[src*='thunderpick.png']"):
        odds_div = img.find_parent("div").find_parent("div")
        break

    text = odds_div.get_text(" ", strip=True)
    winning_team_match = re.search(r"on\s+(.+?)\s+returned", text)
    odd_match = re.search(r"at\s+([0-9]+(?:\.[0-9]+)?)\s+pre-match odds", text)

    winning_team = str(winning_team_match.group(1))
    odd = float(odd_match.group(1))

    implied_p = 1.0 / odd
    inverse_odds = 1.0 / (1.0 - implied_p)

    team_a_odd = odd if team_a == winning_team else inverse_odds
    team_b_odd = odd if team_b == winning_team else inverse_odds

    return {team_a: team_a_odd, team_b: team_b_odd}


if __name__ == "__main__":
    vlr_link = input("Enter the vlr link for the match: ")
    odds = get_match_odds(vlr_link)
    for team, odd in odds.items():
        print(f"{team}: {odd:.2f}")
