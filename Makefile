.PHONY: install clean run

install:
	poetry install
	poetry run playwright install-deps

clean:
	poetry env remove --all || true

model:
	poetry run python3 -m valorant_match_predictor.main

scrape-odds:
	poetry run python3 -m scrapers.match_odds_scraper

scrape-matches:
	poetry run python3 -m scrapers.match_stats_scraper