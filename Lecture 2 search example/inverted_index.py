##
# InvertedIndex
#
# Inverted index.  Keeps track of terms and the documents they occur in
# as well as count (posting lists).
#
# Uses a simple linked list data structure internally.  
#
# I have expanded this to support a basic vector search model as well.  Typically
# an inverted index doesn't do this (as it's just a data structure) but i've included
# the functionality here for demonstration purposes.
#
# Revision History:
# ~~~~~~~~~~~~~~~~~
# 07/10/2019 - Created (CJL).
# 23/04/2021 - Updated to use RE and more thorough pre-processing (CJL).
# 23/04/2021 - Added vector space search functionality (CJL).
# 06/09/2022 - Included space for Log-Entropy calculation (CJL).
###

from linked_list import LinkedList
from linked_list import LinkedListIterator
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import math
import re


class InvertedIndex:
    ##
    # Constructor
    #
    # No parameters required.  Simply creates an empty list of terms, document names
    # as well as loading in english stop words and initialising the Porter stemmer.
    #
    # The index itself is made up of the terms list described below.
    # Each terms list entry consists of an ordered pair.
    # term[i] = [a, b, c]
    #           a - Term indexed
    #           b - Linked list containing list data items
    #           c - Total TF over all documents for term a
    #               Each data item is of the form [c, d, e, f]
    #               c - document identifier
    #               d - Term frequency of term a in document c.
    #               e - This is the TFIDF value if calculated
    #               f - This is the Log-Entropy value if calculated
    # The terms list is kept sorted on the term value, so we can use binary search.
    #
    # The docs list is simply a list of the documentID's passed into
    # the InvertedIndex to be indexed.  This list is also kept sorted.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 07/10/2019 - Created (CJL).
    # 23/04/2021 - Added global stopword list/stemmer object for efficiency (CJL).
    # 23/04/2021 - Added object variable for holding term by doc matrix (possibly scaled)
    #              for performing comparison operations (CJL).
    # 06/09/2022 - Extended list data item to include space for Log-Entropy value (CJL).
    # 07/09/2022 - Extended initial list [x, y, z] to include z which represents the total
    #              term count for term "x" over all documents (CJL).
    ###
    def __init__(self):
        # We need to keep track of the list of terms we are dealing with
        # Each word will be listed as a list
        # [x, y, z] = x is the term
        #             y is a linked list containing lists
        #             z is the total TF for term x over all documents in y
        #             The linked list data item is a list of the form [a, b, c, d]
        #             [a, b, c, d] = a - document identifier, b term frequency of term x
        #                            in document a, c the tf-idf value if calculated,
        #                            d the Log-Entropy value if calculated
        #             z total term count overall for term x
        self.terms = []

        # We need to keep track of the documents we have had added
        # Note, in practice you would use an ordinal value to represent the
        # document, not necessarily a text name.  String comparisons are *slow*.
        self.docs = []

        # Store this once globally in out object rather than recreating it all the time
        # when processing text
        self.stop_words = set(stopwords.words('english'))

        # Create global stemmer for efficiency reasons
        self.stemmer = PorterStemmer()

        # Local variable for holding the term by document matrix for doing
        # calculations.
        self.A = None

    ##
    # Adds a document to the InvertedIndex class.
    #
    # @param document - Ordered pair in the format (a, b)
    #                   a - Document identifier (to go in docs list)
    #                   b - String containing document content to be processed
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 08/10/2019 - Created (CJL).
    # 01/05/2021 - Added binary search support (CJL).
    # 06/09/2022 - Added "None" values for TFIDF/Log-Entropy in ll insertion (CJL).
    # 07/09/2022 - Updated so total term count over all documents for each individual
    #              term is updated (CJL).
    ###
    def add_document(self, document):
        doc_id = document[0]
        text = document[1]

        # Sanity check to ensure unique document being added
        if doc_id in self.docs:
            raise Exception("Document [" + doc_id + "] already exists in the Inverted Index.")

        # Tokenize and process text.  This is where any text pre-processing
        # will take place.
        p_text = self.process_text(text)
        self.docs.append(doc_id)

        # Returns container Counter({'term1' : x_1, 'term2' : x_2, ...})
        # ordered by x_n where x_n is the count of the term termn
        # This way when we iterate to add these to the inverted index we just add
        # a term once with its total count
        counted_text = Counter(p_text)

        # Sorts doc identifiers after every addition, in practice you'd
        # only do this after everything is added if at all
        self.docs.sort()

        # At this point we have our document identifier and a processed
        # list of terms, we can now start inserting this into our postings
        # list LinkedList structure
        for t in counted_text:
            # Extract number of times term t occurs in this document
            c = counted_text[t]
            new_term = False

            # Search for term in our term list...
            # trm - item in terms list if found (Hone otherwise)
            # x - it's index in the terms list if found
            trm, x = self.binary_search_for_term(t)

            # Test if the term is new or already exists
            if trm is None:
                # New term
                new_term = True
                ll = LinkedList()
                ll.insert_at_end([doc_id, c, None, None])
                self.terms.append([t, ll, c])
            else:
                # Postings list already exists, just add onto the end.
                ll = trm[1]
                # Update total term count over all documents
                trm[2] += c
                ll.insert_at_end([doc_id, c, None, None])

            # Any time we add a new term we need to resort the terms
            if new_term is True:
                self.terms.sort(key=lambda tup: tup[0])

    ##
    # This is the method that processes a string of text to be added to
    # the postings list.
    #
    # Any query you want to do would also have to be processed by this method to
    # ensure a "like for like" lexicon for search.
    #
    # @param text - String of text to be added to inverted index
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 08/10/2019 - Created (CJL).
    # 18/04/2021 - Added stop word removal, removal of punctuation/non words,
    #              blank terms and stemming (CJL).
    ###
    def process_text(self, text):
        # We are just going to trivially split on whitespace
        t = text.split()

        # Then we will lowercase everything
        t = [w.lower() for w in t]

        # Stop word removal
        t = [w for w in t if not w in self.stop_words]

        # Get rid of punctuation and non words
        t = [re.sub('[^\w]', "", w) for w in t]

        # If our process above came across something purely not word oriented
        # (for example "--" then we may have empty strings, get rid of those)
        t = [w for w in t if w]

        # Get rid of purely numerical data
        t = [w for w in t if not w.isnumeric()]

        # Last step, stem the terms
        t = [self.stemmer.stem(w) for w in t]

        # Any other text pre-processing wou0ld take place here

        return t

    ##
    # Crude linear search for the presence of a term in the inverted
    # index.  Just traverses the terms list for a match.
    #
    # @param t - Term to search for.
    #
    # @return w - The word data list if found.  Remember this takes the
    #             form [a, b, c] where:
    #             a - Term we were searching for (match)
    #             b - Postings list linked list of documents/TFs.
    #             c - The total TF of a over all documents it appears in.
    #         i - Index of w in the terms array.  Useful for using as an index into
    #             a term by document matrix.
    #         NOTE:  Returns None, None if not found.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 09/10/2019 - Created (CJL).
    ###
    def search_for_term(self, t):
        for w in self.terms:
            if w[0] == t:
                return w, self.terms.index(w)

        return None, None

    ##
    # Binary search for the presence of a term in the inverted index.
    #
    # @param t - Term to search for.
    #
    # @return w - The word list data if found.  Remember this takes the
    #             form [a, b, c] where:
    #             a - Term we were searching for (match)
    #             b - Postings list linked list of documents/TFs.
    #             c - The total TF of a over all documents it appears in.
    #         i - Index of w in the terms array.  Useful for using as an index into
    #             a term by document matrix.
    #         NOTE:  Returns None, None if not found.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 01/05/2021 - Created (CJL).
    # 07/09/2022 - Updated documentation to reflect total TF over all documents is
    #              present int the word data (CJL).
    ###
    def binary_search_for_term(self, t):
        low = 0
        high = len(self.terms) - 1
        mid = 0

        while low <= high:
            # for get integer result
            mid = (high + low) // 2

            # Check if n is present at mid
            if self.terms[mid][0] < t:
                low = mid + 1

                # If n is greater, compare to the right of mid
            elif self.terms[mid][0] > t:
                high = mid - 1

                # If n is smaller, compared to the left of mid
            else:
                return self.terms[mid], mid

                # element was not present in the list
        return None, None

        ##

    # Retrieves a set object that contains the list of document ID's that the
    # term passed is contained in.
    #
    # The sets returned from this function can be used in boolean search operations.
    #
    # @param t - Term we want the document set for (if it exists)
    #
    # @return s - Set containing document ID's for all the documents containing
    #             the term 't'.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 01/05/2021 - Created (CJL).
    ###
    def get_document_set_from_term(self, t):
        # put the term through the same pre-processing all other text goes through
        t = self.process_text(t)

        # Search for the term.  Add document ID's to set to return.
        try:
            ll, x = self.binary_search_for_term(t[0])
            iter = LinkedListIterator(ll[1])
            documents = []
            while True:
                try:
                    item = next(iter)
                    documents.append(item[0])
                except StopIteration:
                    break
        except:
            return set()

        return set(documents)

    ##
    # Retrieves a list that contains the contents of the postings list for a particular
    # term.
    #
    # @param t - Term we want the postings list for (if it exists)
    #
    # @return l - List containing the postings list for term 't' passed.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 02/05/2021 - Created (CJL).
    ###
    def get_postings_list_from_term(self, t):
        # put the term through the same pre-processing all other text goes through
        t = self.process_text(t)

        # Search for the term.  Add contents into list to return.
        try:
            ll, x = self.binary_search_for_term(t[0])
            iter = LinkedListIterator(ll[1])
            list_items = []
            while True:
                try:
                    item = next(iter)
                    list_items.append(item)
                except StopIteration:
                    break
        except:
            return []

        return list_items

    # Returns total number of terms in our inverted index
    def get_total_terms(self):
        return len(self.terms)

    # Returns total number of documents in our inverted index
    def get_total_docs(self):
        return len(self.docs)

    # Returns total TF of term over all documents (0 if it does not exist)
    def get_total_tf(self, t):
        t = self.process_text(t)

        wl, x = self.binary_search_for_term(t[0])
        if x is not None:
            return wl[2]
        else:
            return 0

    # Returns the document frequency of term "t"
    def get_doc_freq(self, t):
        t = self.process_text(t)

        wl, x = self.binary_search_for_term(t[0])

        if x is not None:
            # The length of the posting list is the DF.
            return wl[1].get_size()
        else:
            return 0

    ########################################################################################
    ##
    ##
    ## The functionality below is for vector search space functionality for demonstration
    ## purposes.  Typically an inverted index is purely a data structure and you would not
    ## do this type of thing.  Inverted Indices do support boolean search quite well though.
    ##
    ##
    ####

    ##
    # Search the corpus of documents given a query vector.
    #
    # Basic method is to calculate the cosine similarity between each document and the
    # query vector and return a ranked list of the results.
    #
    # This assumes we have already created the term/doc matrix in the object.
    #
    # @param q - Query vector
    #        tfidf - Flag as to whether or not to use TFIDF search query.
    #                The term/doc matrix should be already generated at this point
    #                and you should use the same setting as that.
    #        log_entropy - Flag as to whether or not to use Log-Entropy search query.
    #                the term/doc matrix should be already generated at this point
    #                and you should use the same setting as that.
    #
    # @return A list of tuples (x, y) where X is the document name and y is the cosine
    #         similarity score.  This is sorted by similarity score.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 23/04/2021 - Created (CJL).
    ###
    def search(self, q, tfidf=False, log_entropy=False):
        # Sanity check flags
        if tfidf and log_entropy:
            raise Exception("[search]:  Error, both tfidf and log_entropy can't both be True.")

        # First, process the query and turn it into an appropriate vector representation
        qv = self.create_query_vector(q, tfidf, log_entropy)

        # Iterate through our corpus doing comparisons
        results = []
        for i in range(self.get_total_docs()):
            v = self.get_doc_vector(i)
            cos = self.cosine_comparison(qv, v)
            results.append((self.docs[i], cos))

        # sort list of tuples on cosine value
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    ##
    # Given the state of the inverted index generate a term by document
    # matrix from its contents.
    #
    # Each row represents a term (sorted) and each column represents a
    # document (sorted on the document identifier).
    #
    # A value in the matrix A, say A[i][j], represents either number of times
    # term i (terms[i]) occurs in document j (docs[j]) or it's TFIDF/Log-Entropy value.
    #
    # To use TFIDF values the calcTFIDF method should be called first so the
    # values are present in the postings list.  Same idea for LogEntropy, call
    # calcLogEntropy first.
    #
    # @param tfidf - Boolean value as to whether or not the matrix created uses
    #                TF-IDF values instead of TF values.  Note, you need to call
    #                the calcTFIDF method first if you want ot use TFIDF values.
    # @param log_entropy - Boolean value as to whether or not the matrix created
    #                uses Log-Entropy values instead of TF values.
    #
    # @return A - Term by document matrix (with TF/TFIDF/LogEntropy values) from
    #             internal inverted index structure.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 08/10/2019 - Created (CJL).
    # 23/04/2021 - Added parameter to optionally build with TFIDF (CJL).
    # 06/09/2022 - Added parameter to optionally build with Log-Entropy (CJL).
    ###
    def generate_term_by_doc_matrix(self, tfidf=False, log_entropy=False):
        # Parameter sanity check
        if tfidf and log_entropy:
            raise Exception("[generate_term_by_doc_matrix]:  Error, both tfidf and log_entropy can't both be True.")

        total_docs = self.get_total_docs()
        total_terms = self.get_total_terms()

        # We need to create a total_terms by total_docs matrix
        A = [[0.0 for i in range(total_docs)] for j in range(total_terms)]

        # Go through terms, then documents present in the postings list to update the
        # A matrix.
        for i in range(total_terms):
            # Get terms postings linked list
            ll = self.terms[i][1]

            # Iterate through linked list to update documents that are relevant to this term
            # Search for the term.
            # i = self.terms.index(t)
            iter = LinkedListIterator(ll)
            while True:
                try:
                    item = next(iter)
                    d = item[0]
                    j = self.docs.index(d)
                    if tfidf is True:
                        A[i][j] = item[2]
                    elif log_entropy is True:
                        A[i][j] = item[3]
                    else:
                        A[i][j] = item[1]
                except StopIteration:
                    break

        # Store this internally for future reference for vector space search
        self.A = A

        return A

    ##
    # Returns document vector from matrix A given a document index.
    #
    # @param doc_ind - Index of column in matrix A we want to get the vector for.
    #
    # @return v - Vector representing document located at index doc_ind.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 30/04/2021 - Created (CJL).
    ###
    def get_doc_vector(self, doc_ind):
        # Populate the vector assuming A has been calculated
        if self.A == None:
            return None

        # We are essentially slicing a vector out that represents a document in the A matrix.
        v = [self.A[i][doc_ind] for i in range(self.get_total_terms())]

        return v

    ##
    # Given a search phrase create a query vector
    #
    # @param q - Query phrase that is unprocessed
    # @param tfidf - Optional parameter to build vector using TFIDF.  Defaults to False.
    # @param log_entropy - Optional parameter to build vector using Log-Entropy.
    #                      Defaults to False.
    #
    # @return - Vector representing the query phrase in the search space.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 23/04/2021 - Created (CJL).
    # 06/09/2022 - Added log_entropy parameter for building query with that (CJL).
    ###
    def create_query_vector(self, q, tfidf=False, log_entropy=False):
        # Parameter sanity check
        if tfidf and log_entropy:
            raise Exception("[create_query_vector]:  Error, both tfidf and log_entropy can't both be True.")

        # First, process the text as per any other text in the index
        q = self.process_text(q)

        # Next, create an empty vector.  It will be 1 X [number of terms]
        v = [0.0 for i in range(self.get_total_terms())]

        # Next, for each term increment its relevant location in the vector
        empty = True
        for t in q:
            # First find its location
            try:
                _, t_ind = self.search_for_term(t)
                v[t_ind] += 1.0
                empty = False
            except:
                # If we get here, it's not in our lexicon
                pass

        # Sanity check, if query vector is empty (that is, there is no matching terms
        # in our lexicon) we should appropriately freak out.
        if empty is True:
            raise Exception("The query vector for [" + str(q) + "] is empty.")

        # We need to take a 2nd pass to update vector as TFIDF if that was selected as
        # some query terms may have occurred more than once which means we can't do it
        # in the first pass
        if tfidf is True:
            total_docs = self.get_total_docs()

            # This is pretty inefficient, should just go through terms that are in query
            for x in range(self.get_total_terms()):
                value = v[x]
                if value > 0.0:
                    ll = self.terms[x][1]
                    docFreq = ll.get_size()
                    invDocFreq = math.log(total_docs / docFreq)
                    v[x] = value * invDocFreq

        # take a 2nd pass to update vector as Log-Entropy if that was selected
        if log_entropy is True:
            # do the same thing as above but with log entropy
            docs = self.get_total_docs()
            for x in range(self.get_total_terms()):
                value = v[x]
                # get term frequency
                term_freq = self.terms[x][2]

                if value > 0.0:
                    ll = self.terms[x][1]
                    iter2 = LinkedListIterator(ll)
                    total_freq = 0
                    while True:
                        try:
                            item2 = next(iter2)
                            total_freq += item2[1]
                        except StopIteration:
                            break
                    # find the proportion of term i in document j with respect to the total frequency count of f(i,j)
                    prop = term_freq / total_freq
                    # find the log entropy
                    log_entropy = math.log(1 + term_freq) * (1 + (prop * math.log(prop)) / math.log(docs))
                    v[x] = value * log_entropy

        return v

    ##
    # Do a cosine comparison between two vectors.  We will assume both vectors are
    # in comparable state.
    #
    # @param v1 - Vector 1
    # @param v2 - Vector 2
    #
    # @return - Cosine similarity score between them.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 23/04/2021 - Created (CJL).
    ###
    def cosine_comparison(self, v1, v2):
        # Normally, some sanity checking should be done here.  For example are both
        # vectors of the same length?  Appropriate type?  Etc.
        n = len(v1)

        # Cosine between two vectors is their dot product divided by the product of magnitudes
        dot_prod = 0.0
        len1 = 0.0
        len2 = 0.0
        for i in range(n):
            dot_prod += v1[i] * v2[i]
            len1 += v1[i] * v1[i]
            len2 += v2[i] * v2[i]

        len1 = math.sqrt(len1)
        len2 = math.sqrt(len2)

        return dot_prod / (len1 * len2)

    ##
    # Calculates the TFIDF values for the current lexicon for future use.
    #
    # Stores the TFIDF value in the postings list as the 3rd item in the list contained
    # for each document.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 23/04/2021 - Created (CJL).
    ###
    def calcTFIDF(self):
        docs = self.get_total_docs()
        terms = self.get_total_terms()

        for t in self.terms:
            # Get list of documents for this term
            ll = t[1]
            docFreq = ll.get_size()
            invDocFreq = math.log(docs / docFreq)
            iter = LinkedListIterator(ll)
            while True:
                try:
                    item = next(iter)
                    tfidf = item[1] * invDocFreq
                    item[2] = tfidf
                except StopIteration:
                    break

    ##
    # Calculates the LogEntropy values for the current lexicon for future use.
    #
    # Stores the LogEntropy value in the postings list as the 4th item in the list contained
    # for each document.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 06/09/2022 - Created (CJL).
    ###
    def calcLogEntropy(self):
        docs = self.get_total_docs()
        terms = self.get_total_terms()

        # Log entropy A(i,j) = log(1 + f(i,j) * (1 + (Sum to j of (p(i,j)*log(p(i,j)))/ log(n))))
        # where f(i,j) is the frequency of term i in document j, p(i,j) is the proportion of term i in document j with
        # respect to the total frequency count of f(i,j) and n is the total number of documents in the corpus.
        # Calculate the log entropy and store it in the postings list 4th item
        for t in self.terms:
            # Get list of documents for this term
            ll = t[1]
            iter = LinkedListIterator(ll)
            while True:
                try:
                    item = next(iter)
                    # find the term frequency
                    term_freq = item[1]
                    # find the total frequency count of f(i,j)
                    total_freq = 0
                    # go through linked list and get total frequency count of f(i,j)
                    iter2 = LinkedListIterator(ll)
                    while True:
                        try:
                            item2 = next(iter2)
                            total_freq += item2[1]
                        except StopIteration:
                            break
                    # find the proportion of term i in document j with respect to the total frequency count of f(i,j)
                    prop = term_freq / total_freq
                    # find the log entropy
                    log_entropy = math.log(1 + term_freq) * (1 + (prop * math.log(prop)) / math.log(docs))
                    # store the log entropy in the postings list 4th item
                    item[3] = log_entropy
                except StopIteration:
                    break

    ### Debugging random junk below

    # Fun function for displaying to the screen what the current
    # term by document matrix would look like in the Inverted Index
    def print_term_by_doc_matrix(self):
        A = self.generate_term_by_doc_matrix()

        print('{0: >9}'.format(''), end='')
        for d in self.docs:
            print('{0: >6}'.format(d), end='')
        print()

        i = 0
        for t in self.terms:
            print('{0: >10}'.format(t[0]), end='')
            j = 0
            for d in self.docs:
                print('{0: .2f}'.format(float(A[i][j])) + " ", end='')
                j += 1
            print()

            i += 1

    # Prints the individual postings lists
    def print(self):
        for t in self.terms:
            print(t)
            llist = t[1]
            print("--> Document list for : " + t[0])
            llist.print_list()


