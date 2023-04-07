from webdriver_manager.chrome import ChromeDriverManager 
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from selenium import webdriver 
from bs4 import BeautifulSoup
from typing import *
import threading, datetime, pandas, time, sys, os 

class GamesReviewScraper:

    def __init__(self, steam_scraper_config : Dict[ str, str ]) -> None:
        self.steam_scraper_config = steam_scraper_config 
        self.webdriver = webdriver.Chrome(
            service = Service(ChromeDriverManager().install())
        )
        self.webdriver.maximize_window()

    def _scrape_chart_games(self) -> List[ str ]:

        try:

            # timeout if page fails to load within specified period 
            self.webdriver.set_page_load_timeout(self.steam_scraper_config["steam_connect_timeout"])
            
            # fetch content from "Steam Chart" (with HTTP/GET)
            self.webdriver.get(self.steam_scraper_config["steam_chart_url"])
        
        except (TimeoutException):

            # return empty list upon timeout  
            return []

        while True:

            # query "Steam Chart" for top-ranked games 
            steam_game_links = self.webdriver.find_elements(By.CLASS_NAME, 
                self.steam_scraper_config["steam_chart_row_class"]
            )
        
            # proceed upon finding games on chart 
            if (len(steam_game_links)):
                break 
            
            # wait for dynamic content to load 
            time.sleep(0.50)

        # extract official links to each game 
        steam_game_links = [
            game_link.find_element(By.CLASS_NAME, 
                self.steam_scraper_config["steam_chart_link_class"]).get_attribute("href")
                    for game_link in steam_game_links 
        ]

        return steam_game_links

    def _to_review_links(self, steam_game_links : List[ str ]) -> List[ str ]:

        target_link = self.steam_scraper_config["steam_review_link_format"]

        game_review_links = []

        for game_link in steam_game_links:

            # serial number of target game [ "https://ABC.com/app/123/DEF" => "123" ]
            game_link = game_link.split("/")[4]

            # positive reviews 
            game_review_links.append(target_link.format(game_link, "positive"))

            # negative reviews 
            game_review_links.append(target_link.format(game_link, "negative"))

        return game_review_links

    def scrape_review_links(self) -> List[ str ]:
        return self._to_review_links(self._scrape_chart_games())

    def _scrape_game_reviews(self, game_link : str) -> List[ Tuple[ str, str ] ]:
        
        print(f"[ Scraping ] [ {game_link} ]")

        def scroll_to_load() -> None:
            self.webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.75);")
            time.sleep(1.50)
            self.webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        try:

            # timeout if page fails to load within specified period 
            self.webdriver.set_page_load_timeout(self.steam_scraper_config["steam_connect_timeout"])
            
            # fetch content from game review page (with HTTP/GET)
            self.webdriver.get(game_link)
        
        except (TimeoutException):

            # return empty list upon timeout 
            return []

        def find_number_of_views():
            nonlocal num_reviews_found, page_content 

            # update reviews # until reaching certain quantity 
            while (num_reviews_found < self.steam_scraper_config["steam_review_minimum_quantity"]):

                # wait until page content is loaded 
                if (page_content is not None):

                    # track number of review posts 
                    num_reviews_found = len(BeautifulSoup(page_content, "lxml").find_all(
                        class_ = self.steam_scraper_config["steam_review_post_class"])
                    )

                time.sleep(1.00)

        # number of reviews discovered 
        num_reviews_found = 0

        # number of reviews previously discovered 
        prev_num_reviews = 0

        # timestamp when "prev_num_reviews" was recorded 
        sot = datetime.datetime.now()

        # content found on webpage 
        page_content = None 

        # thread responsible for tracking number of reviews 
        thread = threading.Thread(target = find_number_of_views)

        thread.start()

        while (num_reviews_found < self.steam_scraper_config["steam_review_minimum_quantity"]):

            # load content by scrolling down 
            scroll_to_load()

            page_content = self.webdriver.page_source

            # terminate searching after review # remained over certain duration 
            if ((datetime.datetime.now() - sot).total_seconds() >= self.steam_scraper_config["steam_review_load_timeout"]):
                if (num_reviews_found == prev_num_reviews):
                    break 
                prev_num_reviews = num_reviews_found 
                sot = datetime.datetime.now()

            sys.stdout.write(f"\r[ Discovered ] [ {num_reviews_found} ]")
            sys.stdout.flush()

            time.sleep(1.20)

        sys.stdout.write(f"\r< {num_reviews_found} Reviews Discovered >")

        print("\n")

        thread.join()

        game_review_elements = BeautifulSoup(page_content, "lxml").find_all(
            class_ = self.steam_scraper_config["steam_review_post_class"]
        )

        scraped_contents = []

        scraped_votings = []

        for game_review in game_review_elements:

            # ignore invalid elements 
            if (game_review is None):
                continue 

            content = game_review.find(
                class_ = self.steam_scraper_config["steam_review_text_class"]
            )

            # ignore invalid elements 
            if (content is None):
                continue 

            content = content.contents

            # concatenate children strings of this element 
            content = "".join(child.strip() for child in content if isinstance(child, str))
            
            # whether user recommends target game { "Recommended", "Not Recommended" }
            voting = game_review.find(class_ = self.steam_scraper_config["steam_review_vote_class"]).text

            # ignore duplicate reviews 
            if (content in scraped_contents):
                continue 

            scraped_contents.append(content)
            
            scraped_votings.append(voting)

        return list(zip(scraped_votings, scraped_contents))

    @staticmethod 
    def save_result_to_csv(filename : str, scrape_content : List[ Tuple[ str, str ] ], *args, **kwargs) -> bool:
            
        try:

            # [ (voting1, review1), ... ] => ([ voting1, ... ], [ review1, ... ])
            scrape_content = tuple(zip(*scrape_content))

            # column names : [ 'voting', 'content' ]
            pandas.DataFrame({
                "voting" : scrape_content[0],
                "content" : scrape_content[1]
            }).to_csv(filename, *args, **kwargs)

            return True 

        except (PermissionError, OSError):

            # signal failure to save file 
            return False 

if (__name__ == "__main__"):

    test = 1

    if (test == 1):

        result_folder = os.path.join(os.path.dirname(__file__), "reviews")
        
        if not (os.path.exists(result_folder)):
            os.makedirs(result_folder)

        steam_scraper_config = {
            "steam_review_link_format" : "https://steamcommunity.com/app/{}/{}reviews/?p=1&browsefilter=toprated",
            "steam_chart_url" : "https://store.steampowered.com/charts/mostplayed",
            "steam_chart_link_class" : "weeklytopsellers_TopChartItem_2C5PJ",
            "steam_chart_row_class" : "weeklytopsellers_TableRow_2-RN6",
            "steam_review_text_class" : "apphub_CardTextContent",
            "steam_review_post_class" : "modalContentLink",
            "steam_review_vote_class" : "title",
            "steam_review_minimum_quantity" : 100,
            "steam_review_load_timeout" : 20,
            "steam_connect_timeout" : 10
        }

        # initialization
        scraper = GamesReviewScraper(steam_scraper_config)

        # scrape review links for various games 
        for url_idx, url in enumerate(scraper.scrape_review_links()):

            # scrape specified number of reviews and save as ".csv"
            if not (scraper.save_result_to_csv(os.path.join(result_folder, f"result_{url_idx}.csv"), 
                scraper._scrape_game_reviews(url), encoding = "utf-8", index = False
            )):
                print(f"Error saving content: {url}")
