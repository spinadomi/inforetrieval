from scholarly import scholarly
import PySimpleGUI as sg
from serpapi import GoogleSearch


# This main file is for the Assignment 2 part of the homework.
def main():
    create_graphic()
    exit()


def get_publication_info(author_name):
    search_query = scholarly.search_author(author_name)
    author = scholarly.fill(next(search_query))
    author_id = author['scholar_id']
    params = {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "api_key": "f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    # count how many articles written for each year by the author
    publication_info = {}
    for article in results["articles"]:
        publication_year = article["year"]
        if publication_year in publication_info:
            publication_info[publication_year] += 1
        else:
            publication_info[publication_year] = 1

    publication_info_array = []
    for year in publication_info:
        publication_info_array.append([year, publication_info[year]])

    # sort publication_info_array by year
    publication_info_array.sort(key=lambda x: x[0])
    return publication_info_array


def get_publisher_info(author_name):
    search_query = scholarly.search_author(author_name)
    author = scholarly.fill(next(search_query))
    author_id = author['scholar_id']
    params = {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "api_key": "f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    # count how many articles were published by each publisher
    publisher_info = {}
    for article in results["articles"]:
        # if article["publication"] is not None:
        if "publication" in article:
            publisher = article["publication"]
            if publisher in publisher_info:
                publisher_info[publisher] += 1
            else:
                publisher_info[publisher] = 1

    publisher_info_array = []
    for publisher in publisher_info:
        publisher_info_array.append([publisher, publisher_info[publisher]])

    return publisher_info_array


def create_graphic():
    sg.theme('DarkAmber')  # Add a touch of color

    headings_publication = ['Publication Year', 'Quantity']
    headings_publisher = ['Publisher', 'Quantity']
    publication_info = []
    publisher_info = []
    layout = [
        [sg.Text('Search Author'), sg.Input(key='-AUTHOR-')],
        [sg.Button('Submit')],
        [sg.Table(values=publication_info, headings=headings_publication, max_col_width=50,
                  auto_size_columns=True,
                  display_row_numbers=True,
                  justification='right',
                  num_rows=20,
                  key='PUBLICATION_TABLE',
                  row_height=15)],

        [sg.Table(values=publisher_info, headings=headings_publisher, max_col_width=50,
                  auto_size_columns=True,
                  display_row_numbers=True,
                  justification='left',
                  num_rows=20,
                  key='PUBLISHER_TABLE',
                  row_height=15)]
    ]
    # Create the Window
    window = sg.Window('GUI for Google Scholar Queries', layout, size=(750, 600))
    publication_table = window['PUBLICATION_TABLE']
    publisher_table = window['PUBLISHER_TABLE']
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
            break
        elif event == 'Submit':
            publication_info = get_publication_info(values['-AUTHOR-'])
            publisher_info = get_publisher_info(values['-AUTHOR-'])
            publication_table.update(values=publication_info)
            publisher_table.update(values=publisher_info)

    window.close()


if __name__ == "__main__":
    main()
