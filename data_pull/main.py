from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
import re
import sys

import util


def parse_tag(tag: BeautifulSoup,
              include_boxes: bool = False,
              verbose: bool = False):

    outlist = []

    # switch statement for tag name
    match tag.name:
        
        # if <a> tag, skip
        case 'a':
            if tag.text == 'Return to text':
                if verbose:
                    print("'Return to text' <a> link. Skipping")
            else:
                print(f"Unidentified <a> tag: {tag}")
        # if <h3> tag, skip
        case 'h3':
            if verbose:
                print("Skipping <h3> tag (excluding section headers)")
            pass
        # if <h4> tag, create new row (subheader)
        case 'h4':
            outlist.append(tag.text)
        # if <h5> tag, possibly a table title to omit
        case 'h5':
            if tag.get("class"):
                # keep if it's a box subheader
                if "subhead1" in tag.get("class"):
                    outlist.append(tag.text)
                else:
                    assert 'tablehead' in tag.get('class')
                    pass
            elif "subsection" in tag.get("id"):
                outlist.append(tag.text)
            else:
                # these used to be unexpected but it seems like they are mainly box subheaders
                if verbose:
                    print(f"Check that this is a box subheader: {tag}")
                outlist.append(tag.text)
        # if <h6> tag, create new row (box subheader)
        case 'h6':
            outlist.append(tag.text)
        # if <p> tag, create new row if it's not a footnote
        case 'p':
            # if it has an id
            if tag.get('id'):
                # if it's a footnote id, skip
                if bool(re.match('f\d+', tag.get('id'))):
                    if verbose:
                        print(f"Footnote <p> tag. Skipping.")
                # if it's another id, flag (unexpected)
                else:
                    print(
                        f"Unexpected <p> id: {tag.get('id')}")

            # else if it's just a non-breaking space, skip
            elif tag.text == '\xa0':
                if verbose:
                    print(f"Empty <p> tag. Skipping.")

            # else if it's the header for references section, skip
            elif tag.text == 'References':
                if verbose:
                    print(f"'References' <p> tag. Skipping.")

            # if none of these special cases, append
            else:
                outlist.append(tag.text)
        # if it's <ul> (bulleted list), then pull each <li> within and process these as separate paragraphs
        case "ul":
            for list_element in tag.find_all(recursive=False):
                outlist.append(list_element.text)
        # if it's <ol> (ordered list), then pull each <li> within and process these as separate paragraphs
        case "ol":
            for list_element in tag.find_all(recursive=False):
                outlist.append(list_element.text)
        # if it's <div> (several sub-options)
        case 'div':

            # if class = 'panel'
            if 'panel' in tag.get('class') and include_boxes:
                
                # pull header and save
                box_header_tag = tag \
                    .find('div', attrs={'class': 'panel-heading'}) \
                    .find(['h5', 'h4'])
                outlist.append(box_header_tag.text)

                # save body using recursive function call
                box_content_tag = tag.find(
                    'div', attrs={'class': 'panel-body'})
                box_outlist = get_paragraph_list(
                    article_tag=box_content_tag,
                    include_boxes=False,
                    verbose=verbose)

                # append to main list
                outlist.extend(box_outlist)

            # if class = 'row', validate it contains an image then skip
            elif 'row' in tag.get('class'):
                # validate that it's a figure, then skip
                assert len(tag.find_all('figure')) == 1
                if verbose:
                    print("Skipping figure")
                pass

            # if class = 'data-table', skip
            elif 'data-table' in tag.get('class'):
                if verbose:
                    print("Skipping data table")
                pass

            # if class = 'footnotes', skip
            elif 'footnotes' in tag.get('class'):
                if verbose:
                    print("Skipping footnotes <div>")
                pass

            # if it's anything else (not expected), flag!
            else:
                if verbose:
                    print(f"Unidentified <div>: {tag}")

        case _:
            print(tag.text)
            print(f"Unexpected tag name: {tag.name}")

    return outlist

def get_paragraph_list(
        article_tag: BeautifulSoup,
        include_boxes: bool = False,
        verbose: bool = False) -> pd.DataFrame:
    """
    Scrape and return the cleaned contents of a section of an FSR.

    This function takes the soup of a given section of an FSR. It then parses this soup,
    creating a structured table where each row is a paragraph of text from the section and its 
    accompanying data. Once complete, it returns this table.

    Parameters
    ----------
        article_tag : BeautifulSoup
            The .html of an FSR section as a BeautifulSoup object. This should not be all the html for the page,
            but just the article tag (`<div id="article">`) and its children tags.
        include_boxes : bool
            Whether to include or omit the content from boxes in this section.
        verbose : bool
            Whether to include verbose output.

    Returns
    -------
        data : pd.DataFrame
            The data for this section as a DataFrame.
    """

    # create empty list of paragraphs
    all_paragraphs = []

    # for each child tag
    for _, child_tag in enumerate(article_tag.find_all(recursive=False)):

        # pull list of all paragraphs within this tag
        tag_paragraphs = parse_tag(tag=child_tag,
                                   include_boxes=include_boxes,
                                   verbose=verbose)

        # extend master list
        all_paragraphs.extend(tag_paragraphs)

    return all_paragraphs

def main(year: str, month: str):
    """
    Scrape and save the text a single domestic FSR from html.

    Note that domestic FSRs are split into different web pages. There is an individual page for each section of the report
    ('Purpose', 'Framework', '1. Asset valuations', etc.).

    This function takes a URL to any one of these pages and does the following:
        - identifies the other URLs to each from the table of contents
        - scrapes/saves data from each page.in order.

    Parameters
    ----------
        - year : str
            The year of report publication, in format YYYY.
        - month : str
            The month of report publication, in format 'month'.
    """
    # create empty data table
    fsr_data = pd.DataFrame(
        columns=util.get_data_columns()
    )

    # identify url of overview section
    domain = 'https://www.federalreserve.gov'

    # may 2021 has a different url than the rest
    if year=="2021" and month=="may":
        endpoint = f'/publications/{month}-{year}-overview.htm'
    else:
        endpoint = f'/publications/{year}-{month}-financial-stability-report-overview.htm'

    # request url, convert to soup
    soup = util.get_soup(url=f"{domain}{endpoint}")

    # get all urls from this page
    urls = util.get_all_urls(soup, domain=domain)

    # identify date
    report_date = dt.datetime.strptime(f'{month} 1 {year}', "%B %d %Y")

    # for each section url
    for url in urls:
        # get soup
        soup = util.get_soup(url=url)

        # pull article tag
        article_tag = soup.find('div', id='article')

        # using url to parse metadata - drop last 4 characters (.htm)
        metadata_str = url.split('/')[-1][:-4]

        # identify section name
        # again it's weird in may 2021
        if year=="2021" and month=="may":            
            section_name_raw = " ".join(metadata_str.split('-')[2:]).lower()
        else:
            section_name_raw = " ".join(metadata_str.split('-')[5:]).lower()
        section_name = util.get_report_section(
            unofficial_name=section_name_raw
        )

        # identify whether to include boxes based on section
        include_boxes = util.should_we_include_boxes(
            report_section=section_name)

        # pull list of paragraphs
        paragraph_list = get_paragraph_list(
            article_tag=article_tag,
            include_boxes=include_boxes,
            verbose=False
        )

        # create and populate dataframe for this section
        section_df = pd.DataFrame(
            columns=util.get_data_columns()
        )
        section_df['body'] = paragraph_list
        section_df['country'] = 'USA'
        section_df['date'] = report_date
        section_df['institution'] = 'Federal Reserve'
        section_df['report_name'] = 'Financial Stability Report'
        section_df['report_section'] = section_name

        # append to master dataframe
        fsr_data = section_df if fsr_data.empty else pd.concat(
            [fsr_data, section_df])

    # add paragraph ids
    fsr_data = fsr_data.reset_index(drop=True)
    ids = "us_fed_fsr_" + \
        fsr_data.date.dt.strftime("%Y%m") + "_" + fsr_data.index.astype(str)
    fsr_data['paragraph_id'] = ids

    # save
    file_path = f"path to folder/us_fed_fsr_{report_date.strftime('%Y%m')}.csv"
    fsr_data.to_csv(
        file_path,
        index=False)
    
    # point user to file
    print(f"Output created, see {file_path}")

if __name__ == '__main__':
    # read command args
    year = sys.argv[1]
    month = sys.argv[2]
    main(year=year, month=month)
