# Information-Retrival-Project
Search engine over the Wikipadia corpus 

Inverted_index_gcp -
Same as the one for HW3, but we had to change the path of one of the read functions.

Backend -
Preprocessing of queries (e.g tokenization), and calculation of TFIDF per search query.

Search_frontend - 
First we read the pickle files from a local folder, this way we save time during the search. We set them as global variables so each method can access them.
Primary search - We will start our solution by analyzing the query, first we will check if the query starts with what or why.
    If the query starts like that, we will use only the functions search title, pagerank and pageviews.
    After that we will check if there are any words that end with s (e.g. plants), and if the last character is not a
    alphabetic letter. 
    Lastly, if the what indicator is 0, we will get the cosim of the title and the body, multiplied by the pagerank 
    and pageviews.

