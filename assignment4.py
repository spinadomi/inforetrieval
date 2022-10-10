import json
from serpapi import GoogleSearch


# api key = f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014
def get_search_results(algo, query, top_n=20):
    if algo == "yahoo":
        params = {
            "engine": "yahoo",
            "p": query,
            "api_key": "f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014"
        }
    else:
        params = {
            "engine": algo,
            "q": query,
            "api_key": "f9b37563533237dd08729f67c653bbed4b86fffe1136be36f7fcc474938e4014"
        }
    # get the results from search
    search = GoogleSearch(params)
    # only get the 'top_n' results
    ranking = search.get_dict()["organic_results"][:top_n]
    return ranking


def main():
    search_query = 'information retrieval evaluation'
    search_results = {
        'google': get_search_results('google', search_query, top_n=7),
        'bing': get_search_results('bing', search_query),
        'duckduckgo': get_search_results('duckduckgo', search_query),
        'yahoo': get_search_results('yahoo', search_query)
    }
    # save search results in a json file
    with open('search_results.json', 'w') as f:
        json.dump(search_results, f, indent=4)

    # get the relevant documents
    relevant_docs = set()
    for result in search_results['google']:
        relevant_docs.add(result['link'])

    # calculate precision and recall
    for algo in search_results:
        retrieved_docs = set()
        for result in search_results[algo]:
            retrieved_docs.add(result['link'])
        precision, recall = precision_recall(retrieved_docs, relevant_docs)
        print(f"{algo} - Precision: {precision}, Recall: {recall}")

    # plot precision at 11 standard recall levels
    for algo in search_results:
        retrieved_docs = []
        for result in search_results[algo]:
            retrieved_docs.append(result['link'])
        precision_levels = precision_at_11_standard_recall_levels(retrieved_docs, relevant_docs)
        print(f"{algo} - Precision at 11 standard recall levels: {precision_levels}")

    # calculate single valued summary
    for algo in search_results:
        retrieved_docs = []
        for result in search_results[algo]:
            retrieved_docs.append(result['link'])
        p_at_5, p_at_10, f_measure = single_valued_summary(retrieved_docs, relevant_docs)
        print(f"{algo} - P@5: {p_at_5}, P@10: {p_at_10}, F-measure: {f_measure}")


def precision_recall(retrieved_docs, relevant_docs):
    # calculate precision and recall
    precision = len(relevant_docs.intersection(retrieved_docs)) / len(retrieved_docs)
    recall = len(relevant_docs.intersection(retrieved_docs)) / len(relevant_docs)
    return precision, recall


def precision_at_11_standard_recall_levels(retrieved_docs, relevant_docs):
    # calculate the precision at 11 standard recall levels, return r_values and p_values
    r_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p_values = []
    for r in r_values:
        # calculate precision at recall level r, make sure not to divide by 0
        if len(retrieved_docs[:int(r * len(relevant_docs))]) == 0:
            p_values.append(0)
        else:
            precision = len(relevant_docs.intersection(retrieved_docs[:int(r * len(relevant_docs))])) / len(
                retrieved_docs[:int(r * len(relevant_docs))])
            p_values.append(precision)
    return r_values, p_values


def single_valued_summary(retrieved_docs, relevant_docs):
    # calculate metric precision at rank 5 (P@5), rank 10 (P@10), and the F-measure
    p_at_5 = len(relevant_docs.intersection(retrieved_docs[:5])) / len(retrieved_docs[:5])
    p_at_10 = len(relevant_docs.intersection(retrieved_docs[:10])) / len(retrieved_docs[:10])
    f_measure = 2 * p_at_5 * p_at_10 / (p_at_5 + p_at_10)
    return p_at_5, p_at_10, f_measure


if __name__ == "__main__":
    main()
