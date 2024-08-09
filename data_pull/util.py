from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
import requests


def get_all_urls(soup: BeautifulSoup, domain: str) -> tuple:
    """
    Retrieve the URL to each section of an FSR from the Soup of a single one.

    For a single FSR, each section page contains a table of contents with links to other sections.
    This function scrapes and returns all the links from this table of contents.

    Parameters
    ----------
        domain : str
            The domain.
        soup : BeautifulSoup
            The .html of an FSR section as a BeautifulSoup object.

    Returns
    -------
        urls : tuple
            The URL for each section of this report as a string, in order.
    """

    # pull table of contents tag
    table_of_contents_tag = soup.find('div', id='t4_nav')
    link_tags = table_of_contents_tag.find_all('a')
    urls = [f"{domain}{link_tag.get('href')}" for link_tag in link_tags]

    return urls


def get_data_columns():
    return ('country', 'date', 'institution', 'report_name', 'report_section', 'paragraph_id', 'body')


def get_new_row(date: dt.datetime, report_section: str, body: str):

    # validate section name
    valid_sections = [
        'Purpose/framework',
        'Overview',
        'Asset valuations',
        'Nonfinancial leverage',
        'Financial leverage',
        'Funding risk',
        'Near-term risks'
    ]
    assert report_section in valid_sections

    # create df
    new_row = pd.DataFrame(
        columns=get_data_columns(),
        data=[['USA', date, 'Federal Reserve',
               'Financial Stability Report', report_section, None, body]]
    )

    return new_row


def get_soup(url: str):
    """Get a BeautifulSoup object for a webpage."""

    https_proxy = 'proxy'
    request = requests.get(
        url,
        proxies={'https': https_proxy}
    )
    soup = BeautifulSoup(
        request.content,
        features='html.parser'
    )

    return soup


def get_report_section(unofficial_name: str):
    unofficial2official = {
        'purpose and framework': 'Purpose/framework',
        'purpose': 'Purpose/framework',
        'framework': 'Purpose/framework',
        'overview': 'Overview',
        'asset valuation': 'Asset valuations',
        'asset valuations': 'Asset valuations',
        'asset valuation pressures': 'Asset valuations',
        'borrowing': 'Nonfinancial sector leverage',
        'borrowing by businesses and households': 'Nonfinancial sector leverage',
        'leverage': 'Financial sector leverage',
        'leverage in the financial sector': 'Financial sector leverage',
        'funding': 'Funding risks',
        'funding risk': 'Funding risks',
        'funding risks': 'Funding risks',
        'near term risks': 'Near-term risks',
        'near term risks to the financial system': 'Near-term risks'
    }
    return unofficial2official[unofficial_name]


def should_we_include_boxes(report_section: str):
    map = {
        'Purpose/framework': False,
        'Overview': False,
        'Asset valuations': True,
        'Nonfinancial sector leverage': True,
        'Financial sector leverage': True,
        'Funding risks': True,
        'Near-term risks': True,
    }
    return map[report_section]
# def get_abbr_report_section(official_name: str):
#     official2abbr = {
#         'Purpose/framework': 'pw',
#         'Funding risk': 'fr',
#     }
#     return official2abbr[official_name]
