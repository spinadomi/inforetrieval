from scholarly import scholarly

search_query = scholarly.search_author('George Azzopardi')
author = scholarly.fill(next(search_query))
print(author)
print([pub['bib']['title'] for pub in author['publications']])