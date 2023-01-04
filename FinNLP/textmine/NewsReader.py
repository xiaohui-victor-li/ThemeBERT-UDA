# Script to parse and collect news text data
# ----------------------------------------
#

import pandas as pd
import time
# HTML parsing
import re
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By

# exceptions
from selenium.common.exceptions import StaleElementReferenceException,TimeoutException

from FinNLP._utils import NLPLogger


WSJ_STOP_CATEGORY_SET = ['PEPPER & SALT', 'CROSSWORD', 'ACROSTIC', 'MLB',
                         'BOOKSHELF', 'OBITUARIES', 'PERSONAL FINANCE', 'FASHION', 'SLIDESHOW']


class WSJnewsReader:
    """NewsReader load and collect the news data from website
    using selenium chrome driver

    Core Functions:
        @collect_news_list: get the news list at specific date
        @get_news_piece: get the news content splitted by '\n\n'
    """
    def __init__(self, level="DEBUG", max_time_out=180):
        # selenium driver
        # this takes some time to launch
        self.driver = webdriver.Chrome() 
        self.driver.set_page_load_timeout(max_time_out)   # wait for driver response before throwing exception
        self.logger = NLPLogger(level, name="FinNLP.WSJNews")
        self.logging_status = False

    def login(self, username, pwd):
        """automatically login"""
        # 1. enter the news 
        self.driver.get("https://www.wsj.com/news/archive")
        self.logger.info(" => start 📰 WSJ login process")
        sign_ls = self.driver.find_elements(By.XPATH,'//a[normalize-space()="Sign In"]')
        sign_ls[-1].click() 
        # 2. input username
        user_input = self.driver.find_element(By.CLASS_NAME, "username")
        user_input.send_keys(username)
        # next page
        user_button = self.driver.find_element(By.XPATH, '//button[normalize-space()="Continue With Password"]')
        user_button.click()
        time.sleep(1)
        # input password
        user_pwd_ls = self.driver.find_elements(By.XPATH, '//input[@class="password" and @name="password"]')
        while len(user_pwd_ls)<1:
            user_pwd_ls = self.driver.find_elements(By.XPATH, '//input[@class="password" and @name="password"]')
        user_pwd_ls[-1].send_keys(pwd)
        login_button_ls = self.driver.find_elements(By.XPATH,
                                    '//button[normalize-space()="Sign In"]')
        login_button_ls[-1].click() 
        self.logger.info(" <= WSJ logging is successful! -- user:{} pwd:{}".format(username, pwd))
        self.logging_status = {'user':username, 'pwd':pwd}


    def collect_news_list(self, input_date):
        """collect the WSJ news list given the date

        RETURN:
            a dataframe of the WSJ news list
        """
        news_web_page = "https://www.wsj.com/news/archive/"+pd.to_datetime(input_date).strftime("%Y/%m/%d")
        G_driver = self.driver
        G_driver.get(news_web_page)
        
        page_pointer_ls = G_driver.find_elements(By.CLASS_NAME, "WSJTheme--SimplePaginator__center--1zmFPX8Z ")
        if len(page_pointer_ls)>0:
            page_pointer = page_pointer_ls[0]
            page_pointer_txt = page_pointer.text
            tot_page = int(page_pointer_txt.split("\n")[-1].split(' ')[-1])
        else:
            # CASE of empty news as reflected in the Archive
            self.logger.info(" => WSJ news as of @{} is empty 🚨".format(input_date))
            return 
        self.logger.info(" => collecting WSJ news list📰 of pages [{}] --@{}".format(tot_page,input_date))

        news_table_df_ls = []
        news_items = ['Category', 'Headline', 'Time']
        for i in range(tot_page):
            try:
                news_source_ls = G_driver.find_elements(By.CLASS_NAME,"WSJTheme--overflow-hidden--qJmlzHgO ")
            except StaleElementReferenceException as e:
                time.sleep(0.1)
                news_source_ls = G_driver.find_elements(By.CLASS_NAME,"WSJTheme--overflow-hidden--qJmlzHgO ")
            
            while len(news_source_ls)==0:
                time.sleep(0.1)
                news_source_ls = G_driver.find_elements(By.CLASS_NAME,"WSJTheme--overflow-hidden--qJmlzHgO ")

            # content
            try:
                content_ls = [x.text for x in news_source_ls]
            except StaleElementReferenceException as e:
                time.sleep(0.1)  # timeout for response
                news_source_ls = G_driver.find_elements(By.CLASS_NAME,"WSJTheme--overflow-hidden--qJmlzHgO ")
            # CASE of empty contents
            if len(content_ls)==0:
                break

            # link
            link_ls = [self._get_news_link(x) for x in news_source_ls]
            while len(link_ls) != len(content_ls):  # case when link list is sparse
                self.logger.info(" ==> [Retrieve] Current Link list: {} < Content list: {}".format(len(link_ls),len(content_ls) ))
                news_source_ls = G_driver.find_elements(By.CLASS_NAME,"WSJTheme--overflow-hidden--qJmlzHgO ")
                link_ls = [self._get_news_link(x) for x in news_source_ls]
                content_ls = [x.text for x in news_source_ls]

            # Combine into news table
            WSJ_news_table = pd.DataFrame([y.split('\n') for y in content_ls])
            WSJ_news_table.columns = news_items
            WSJ_news_table['Link'] = link_ls 
            news_table_df_ls.append(WSJ_news_table)

            # click on next page
            next_page_pointer_ls =  G_driver.find_elements(By.CLASS_NAME,
                                                           "WSJTheme--SimplePaginator__label--3OjlQ93n ")
            if len(next_page_pointer_ls)>0:
                try:
                    next_page_pointer_ls[-1].click()
                    self.logger.info(" ==> Turning to page {}".format(i+2))
                except TimeoutException as e:
                    # CASE of too long waiting time for Next Page
                    break
            else:
                break
           
        tot_news_table = pd.concat(news_table_df_ls)
        tot_news_table['DATE'] = input_date
        return tot_news_table[['DATE', 'Category', 'Headline', 'Time', 'Link']]

    def get_news_piece(self, url, time_sleep_secs=0.1):
        """get the news content splitted by '\n\n'
        """
        G_driver = self.driver
        G_driver.get(url)   # open the url
        
        # subtitle
        subtitle_ls = G_driver.find_elements(By.XPATH, "//h2")

        if len(subtitle_ls)==0:
            subtitle_txt = None
        else:
            subtitle_txt = subtitle_ls[0].text 
        time.sleep(time_sleep_secs)   # wait for loading

        # query all the news paragraphs
        article_ls = G_driver.find_elements(By.CLASS_NAME, "article-content  ")
        if len(article_ls)>0:
            article = article_ls[0]
            # sub paragraphs within the list
            txt_doc_ls = [x.text for x in article.find_elements(By.XPATH, ".//p")]
        else:
            para_ls = G_driver.find_elements(By.XPATH, "//p[@class='css-xbvutc-Paragraph e3t0jlg0']")
            txt_doc_ls = [x.text for x in para_ls]
        # filter out the single item paragraph as noises
        txt_doc_ls = [y for y in txt_doc_ls if len(y.split(' '))>2]
        output_doc = "\n\n".join(txt_doc_ls)

        output_doc_res = {'Subtitle':subtitle_txt, 'Content':output_doc}
        
        return output_doc_res



    def _get_news_link(self, x):
        try:
            news_headline = x.find_element(By.CLASS_NAME, "WSJTheme--headline--7VCzo7Ay ")
            news_link = news_headline.find_element(By.TAG_NAME, 'a').get_attribute('href') 
        except StaleElementReferenceException as e:
            time.sleep(0.05)
            news_headline = x.find_element(By.CLASS_NAME, "WSJTheme--headline--7VCzo7Ay ")
            news_link = news_headline.find_element(By.TAG_NAME, 'a').get_attribute('href') 

        return news_link 

    def close(self):
        self.driver.close()







