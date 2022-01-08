# imports
import time
from flask import Flask, request, jsonify
import backend
from inverted_index_gcp import *

global id_title
where_are_my_little_pickle = '/content/gdrive/MyDrive/postings_gcp/id_title_dict.pickle'  # on the cloud, jesus you horny bastereds
with open(where_are_my_little_pickle, 'rb') as f:
    id_title = dict(pickle.loads(f.read()))

global pageviews
pickle_rick = '/content/gdrive/MyDrive/postings_gcp/pageviews-202108-user.pkl'  # I turned myself into a pickle morty
# https://www.youtube.com/watch?v=8RxDVdP2TZ8&ab_channel=smacksmashen
with open(pickle_rick, 'rb') as f:
    pageviews = dict(pickle.loads(f.read()))

global pagerank
history_channel = '/content/gdrive/MyDrive/postings_gcp/page_rank_1.pkl'
# https://www.history.com/news/pickles-history-timeline
with open(history_channel, 'rb') as f:
    pagerank = dict(pickle.loads(f.read()))

global i_anchor
inverted_anchor = InvertedIndex()
i_anchor = inverted_anchor.read_index('/content/gdrive/MyDrive/postings_gcp', 'anchor_index_2')

global i_text
inverted_text = InvertedIndex()
i_text = inverted_text.read_index('/content/gdrive/MyDrive/postings_gcp', 'text_index_2')

global i_title
inverted_title = InvertedIndex()
i_title = inverted_title.read_index('/content/gdrive/MyDrive/postings_gcp', 'title_index_2')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    '''
    We will start our solution by analyzing the query, first we will check if the query starts with what or why.
    If the query starts like that, we will use only the functions search title, pagerank and pageviews.
    After that we will check if there are any words that end with s (e.g. plants), and if the last character is not a
    alphabetic letter. 
    Lastly, if the what indicator is 0, we will get the cosim of the title and the body, multiplied by the pagerank 
    and pageviews.
    '''
    # query analyze
    ABC = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
           'v', 'b', 'n', 'm']
    first_word = query.split()[0].lower()

    what = 0
    if first_word in ['what', 'why']:
        what = 1

    pre_body = backend.Preprocess()
    pre_body.titles = id_title

    query_to_send = []
    for word in query.split():
        if word[-1].lower() == 's':
            query_to_send.append(word[:-1])
        if word[-1] not in ABC:
            query_to_send.append(word[:-1])
        query_to_send.append(word)
    query = ''
    for word in query_to_send:
        query += word + ' '

    # title search
    title = defaultdict(int)
    for token in list(set(query.split())):
        try:
            pl = i_title.read_posting_list(token.lower())
        except:
            continue
        for doc_id, tf in pl:
            title[doc_id] += 1
    temp_title = list(title.items())
    temp_title_d = dict(temp_title)

    if what == 1:
        for i in temp_title_d:
            try:
                temp_title_d[i] = temp_title_d[i] * pagerank[i]
            except:
                pass
            try:
                temp_title_d[i] = temp_title_d[i] * pageviews[i]
            except:
                pass
            temp = temp_title_d.items()
            temp = sorted(temp, key=lambda x: x[1], reverse=True)[:100]
            res = list(map(lambda x: (x[0], pre_body.titles[x[0]]), temp))
            return jsonify(res)

    # w
    W_title = 1000
    W_text = 0.01
    W_not = 0.001
    result = defaultdict(int)

    # cosim title
    pre_title = backend.Preprocess()
    pre_title.remove_stopwords(query)
    pre_title.tf_idf(i_title, len((i_title.d_len.keys())))
    pre_title.calc_tfIdf(query, i_title)
    temp_title_cosim = pre_title.cos_sin[query]
    temp_title_cosim_d = dict(temp_title_cosim)

    # cosim body
    pre_body = backend.Preprocess()
    pre_body.titles = id_title
    pre_body.remove_stopwords(query)
    pre_body.tf_idf(i_text, len(i_text.d_len.keys()))
    pre_body.calc_tfIdf(query, i_text)
    temp_body = pre_body.cos_sin[query]
    temp_body_d = dict(temp_body)

    # combination
    union = set(temp_body_d.keys()) & set(temp_title_d.keys()) & set(temp_title_cosim_d)
    for i in union:
        temp_title_d[i] = temp_title_d[i] * W_title / len(pre_body.titles[i])
        temp_body_d[i] = temp_body_d[i] * W_text
        temp_title_cosim_d[i] = temp_title_cosim_d[i] * W_title / len(pre_body.titles[i])
        result[i] += (temp_body_d[i] + temp_title_d[i] + temp_title_cosim_d[i])
        try:
            result[i] = result[i] * pagerank[i]
        except:
            pass
        try:
            result[i] = result[i] * pageviews[i]
        except:
            pass
    for i in set(temp_body_d.keys()).symmetric_difference(set(temp_title_d.keys())):
        try:
            result[i] = temp_body_d[i] * W_not
        except:
            result[i] += temp_title_d[i] * W_not

    temp = result.items()
    temp = sorted(temp, key=lambda x: x[1], reverse=True)[:100]
    res = list(map(lambda x: (x[0], pre_body.titles[x[0]]), temp))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    preproc = backend.Preprocess()
    preproc.titles = id_title
    preproc.remove_stopwords(query)
    preproc.tf_idf(i_text, len(i_text.d_len.keys()))
    preproc.calc_tfIdf(query, i_text)

    res = preproc.cos_sin
    res = list(res.items())[0][1][:100]
    res = list(map(lambda x: (x[0], preproc.titles[x[0]]), res))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    pre_title = backend.Preprocess()
    title_dict = defaultdict(int)
    query = pre_title.tokenizer(query).split()
    query_set = list(set(query))
    for token in query_set:
        try:
            pl = i_title.read_posting_list(token.lower())
        except:
            continue
        for doc_id, _ in pl:
            title_dict[doc_id] += 1
    res = list(title_dict.items())
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], id_title[x[0]]), res))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    for i in wiki_ids:
        try:
            res.append(pagerank[i])
        except:
            continue

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """


    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    prep_anchor = backend.Preprocess()
    anchor_dict = defaultdict(int)
    query = prep_anchor.tokenizer(query).split()
    query_set = list(set(query))
    for token in query_set:
        try:
            posting = i_anchor.read_posting_list(token.lower())
        except:
            continue
        for doc_id, _ in posting:
            anchor_dict[doc_id] += 1
    res = list(anchor_dict.items())
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], id_title[x[0]]), res))

    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    for i in wiki_ids:
        try:
            res.append(pageviews[i])
        except:
            continue
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
