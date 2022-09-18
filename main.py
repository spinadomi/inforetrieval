from scholarly import scholarly
import PySimpleGUI as sg
from serpapi import GoogleSearch


# api key = f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014
def main():
    create_graphic()
    exit()


def get_author_info(author_name, sort_method):
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
    articles = results["articles"]
    author_info = []
    for article in articles:
        title = article["title"]
        author = article["authors"]
        publication_year = article["year"]
        citation = article["cited_by"]["value"]
        author_info.append([title, author, publication_year, citation])

    # sort author_info by publication year
    if sort_method == "Publication Year":
        author_info.sort(key=lambda x: x[2])
    # sort author_info by citation
    elif sort_method == "Citation":
        author_info.sort(key=lambda x: x[3])

    return author_info


def create_graphic():
    sg.theme('DarkAmber')  # Add a touch of color

    headings = ['Title', 'Authors', 'Publication year', 'Citations']
    author_info_array = []
    layout = [
        [sg.Text('Search Author'), sg.Input(key='-AUTHOR-')],
        [sg.Text('Select a sort method'), sg.Combo(['Publication Year', 'Citation'], key='-SORT_METHOD-')],
        [sg.Button('Submit')],
        [sg.Table(values=author_info_array, headings=headings, max_col_width=50,
                  auto_size_columns=True,
                  display_row_numbers=True,
                  justification='right',
                  num_rows=40,
                  key='TABLE',
                  row_height=35,
                  tooltip='Author Info Table')]
    ]
    # Create the Window
    window = sg.Window('GUI for Google Scholar Queries', layout, size=(750, 600))
    table = window['TABLE']
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
            break
        elif event == 'Submit':
            author_info = get_author_info(values['-AUTHOR-'], values['-SORT_METHOD-'])
            table.update(values=author_info)

    window.close()


if __name__ == "__main__":
    main()
