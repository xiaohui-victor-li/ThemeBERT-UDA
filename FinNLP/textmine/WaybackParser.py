# Script to query news text content using WayBack
# -----------------------------------------------
# -latest 12/17/2022; 
# 
#

import wayback
import joblib
# Exception Handling
from wayback.exceptions import MementoPlaybackError
from requests.exceptions import ChunkedEncodingError
from bs4 import BeautifulSoup

class WaybackTimeWalker:
    """Wayback Machine🚶 class to parse the snapshot archived website

    Core Functions:
        @collect_news_list: get the news list at specific date
        @get_news_piece: get the news content splitted by '\n\n'
    """
    def __init__(self):
        # WayBack driver
        self.driver = wayback.WaybackClient()
        # cache system to store the parsed content
        self.archive_cache = {}
    
    def search_website(self, url, mode='first', verbosed=True):
        """search & download the achived website given url
        """
        # -> Search archive.org's CDX API for all captures of a given URL. 
        ## Cache check
        if url in self.archive_cache:
            return self.archive_cache[url]

        ## Search new websites
        flag_query = False
        for i,record in enumerate(self.driver.search(url)):
            if mode=='first' and flag_query:  # only keep the first quried website
                self.archive_cache[url] = memento
                break 
            try:
                # Load the mementos
                memento = self.driver.get_memento(record)
                if verbosed:
                    print(" => Find&Collect the {}th website in Wayback {} --@{}".format(i, url,
                                                                                        record.timestamp))
                # 200 OK status indicate valid website
                flag_query = memento.status_code == 200

            except MementoPlaybackError as e:
                continue
            except ChunkedEncodingError as e:
                continue
                
        return memento
            
            

# function to parse the HTML content from WSJ news
def WSJ_html_parse(news_content):
    # apply BeautifulSoup
    news_soup = BeautifulSoup(news_content, 'html.parser')

    # check the availability
    div_article_ls = news_soup.find_all('article')
    if len(div_article_ls)==0:
        # website not captured
        output_res = {'category':'', 'headline':'', 'content':'', 'doc_len':0}
        return output_res
    else:
        div_article = div_article_ls[0]

    full_nav_ls = news_soup.find_all('header', attrs={'class':'article_header'})
    css_nav_ls = news_soup.find_all('div', attrs={'class':'css-j6808u'})
    front_nav_ls = news_soup.find_all('div', attrs={'class':'e10dsoc80'})
    left_nav_ls = news_soup.find_all('div', attrs={'class':'css-1ieir01-HeadlineContainer'})
    # /website with Media feature
    media_nav_ls = news_soup.find_all('div', attrs={'class':'bigTop__text'})  
    if len(media_nav_ls)==0:
        media_nav_ls = news_soup.find_all('div', attrs={'class':'bigTop__media'})
        media_nav_ls = [x for x in  media_nav_ls if x.find_next_sibling('h1')]


    # Different Types website
    if len(media_nav_ls)>0 and media_nav_ls[-1].find_next('h1'):
        raw_category_ls = []
        div_headline = media_nav_ls[-1]
        raw_content_button = div_article.find_next('div', attrs={'class':'wsj-snippet-body'})
        if raw_content_button is None:
            raw_content_button = div_article.find_next('div', attrs={'class':'article-content'})

        raw_content_ls = raw_content_button.text.split('\n\n')

    elif len(full_nav_ls)>0:
        div_headline = full_nav_ls[0]
        raw_category_ls = div_headline.find_all('li', attrs={'class':'article-breadCrumb'}) 
        div_body = div_article.find_next('div',  attrs={'class':'article-content'}) 
        div_body = div_body or div_article.find_next('div',  attrs={'class':'wsj-snippet-body'}) 
        if div_body is not None:
            raw_content_ls = [x.text for x in div_body.find_all('p')]
        else:
            raw_content_ls = []
    elif len(front_nav_ls)>0:
        raw_category_ls = []
        div_headline = front_nav_ls[0]
        raw_content_ls = [x.text for x in div_article.find_all("p", attrs={"data-type":"paragraph"})]
    elif len(left_nav_ls)>0:
        raw_category_ls = []
        div_headline = left_nav_ls[0]
        raw_content_ls = [x.text for x in div_article.find_all("p", attrs={"data-type":"paragraph"})]
    elif len(css_nav_ls)>0:
        div_headline = css_nav_ls[0]
        raw_category_ls = div_headline.find_all('li', attrs={'class':'e1blx2ob3'})
        raw_content_ls = [x.text for x in div_article.find_all("p", attrs={"data-type":"paragraph"})]
    else:
        # /Category
        raw_category_ls = div_article.find_all('li', attrs={'class':'article-breadCrumb'})
        # /Headline
        # headline wrapper
        div_headline_ls = div_article.find_all('div', attrs={'class':'wsj-article-headline-wrap'})
        if len(div_headline_ls)>0:
            div_headline = div_headline_ls[0]
        else:
            div_headline = None
        # /Content
        raw_content_button = div_article.find_next('div', attrs={'class':'wsj-snippet-body'})
        if raw_content_button is not None:
            raw_content_ls = raw_content_button.text.split('\n\n')
        else:
            raw_content_ls = [x.text for x in div_article.find_all("p", attrs={"data-type":"paragraph"})]

    # /Category
    if len(raw_category_ls)>0:
        category_ls = [x.text.replace('\n','').replace(' ','') for x in raw_category_ls]
        category = ' | '.join(category_ls)
    else:
        category = ''
        
    # /Headline
    if div_headline and div_headline.find_next('h1') is not None:
        title = div_headline.find_next('h1').text.replace('\n', '').replace('  ', '')
        # / subtitle
        subtitle_button =div_headline.find_next('h2')
        if subtitle_button is None:
            headline = title.replace('  ',' ').replace('\t', '')
        else:
            subtitle = div_headline.find_next('h2').text.replace('\n', '').replace('  ', '')
            headline = (title+' | '+subtitle).replace('  ',' ').replace('\t', '')
    else:
        headline = ''

    # /Content
    content_ls = []
    for para in raw_content_ls:
        process_text = para.replace('\n',' ').replace('  ',' ').replace('... ', '')
        content_ls.append(process_text)
    content = '\n\n'.join(content_ls).replace('  ', '')

    # /Doc Length
    doc_len = len(content.split(' '))

    # wrap up output
    output_res = {'category':category, 'headline':headline, 'content':content, 'doc_len':doc_len}

    return output_res 



