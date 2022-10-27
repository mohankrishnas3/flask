from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask import request
import threading
import logging
from flask.logging import default_handler
import json
import requests


from rake_nltk import Rake

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from flask_cors import cross_origin

import numpy as np
import itertools

import bs4 as bs
import urllib.request
import re
import nltk


def summarize_nltk(input_sentence):
    import bs4 as bs
    import urllib.request
    import re
    import nltk

    scraped_data = urllib.request.urlopen(
        "https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome_coronavirus_2"
    )
    article = scraped_data.read()
    # article = '''
    # It is actually funny when you think of it, liberals and democrats hated the Georgia voting bill they never read so much they forced the all star game moved to Colorado where it is actually harder to vote, you must have an ID to mail in ballots, and less early voting days than Georgia or Texas...the Georgia and Texas law simply made it harder to cheat which democrats and liberals hate. Because if it was about voting they would be protesting democrat states that actually have the most strict voting laws in the United States!
    # '''
    # article = '''
    # The channel was created by Australian-American media mogul Rupert Murdoch to appeal to a conservative audience, hiring former Republican media consultant and CNBC executive Roger Ailes as its founding CEO. It launched on October 7, 1996, to 17 million cable subscribers. Fox News grew during the late 1990s and 2000s to become the dominant subscription news network in the U.S. As of September 2018, approximately 87,118,000 U.S. households (90.8% of television subscribers) received Fox News. In 2019, Fox News was the top-rated cable network, averaging 2.5 million viewers. Murdoch is the current executive chairman and Suzanne Scott is the CEO.
    # '''
    article = input_sentence

    parsed_article = bs.BeautifulSoup(article, "lxml")

    paragraphs = parsed_article.find_all("p")

    article_text = input_sentence

    for p in paragraphs:
        article_text += p.text
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r"\[[0-9]*\]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)
    # Removing special characters and digits
    formatted_article_text = re.sub("[^a-zA-Z]", " ", article_text)
    formatted_article_text = re.sub(r"\s+", " ", formatted_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words("english")

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequncy
        sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(" ")) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    import heapq

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = " ".join(summary_sentences)

    # print("input sentence ")
    # print(article)

    # print("input sentence after summarization using NLTK package")
    # print(summary)
    return summary


def summarize_bert(sentence):
    # article = sentence
    # from summarizer import Summarizer
    # model = Summarizer()
    # result = model(article, min_length= 1)
    # #result = model(article, ratio=0.2)  # Specified with ratio
    # #result = model(article, num_sentences=3)
    # summary_bert = "".join(result)
    # # return summary_bert
    # from transformers import pipeline
    # summarizer = pipeline("summarization", model = "bert-base-uncased")
    # return summarizer(sentence)[0]['summary_text']
    from summarizer.sbert import SBertSummarizer

    model = SBertSummarizer("paraphrase-MiniLM-L6-v2")
    result = model(sentence, num_sentences=1000)
    return result


def extract_keywords_using_rake(input_sentence):
    summary = input_sentence
    from rake_nltk import Rake

    rake_nltk_var = Rake()
    # text = """spaCy is an open-source software library for advanced natural language processing,
    # written in the programming languages Python and Cython. The library is published under the MIT license
    # and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
    rake_nltk_var.extract_keywords_from_text(summary)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    # print("\n keywords extracted after summarization using RAKE Algorithm")
    # print(keyword_extracted)
    return keyword_extracted


def extract_keywords_using_distil_bert(
    input_sentence, number_of_words_in_each_keyword, number_of_top_keywords
):
    summary = input_sentence
    from sklearn.feature_extraction.text import CountVectorizer

    n_gram_range = (number_of_words_in_each_keyword, number_of_words_in_each_keyword)
    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
        [summary]
    )
    candidates = count.get_feature_names()
    candidates
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    doc_embedding = model.encode([summary])
    candidate_embeddings = model.encode(candidates)

    from sklearn.metrics.pairwise import cosine_similarity

    top_n = number_of_top_keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return keywords


# max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
# def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
# def mmr(doc_embedding, word_embeddings, words, top_n, diversity):


def bert_keywords_diversificatin_using_max_sum_similarity_method(
    input_sentence,
    number_of_words_in_each_keyword,
    number_of_top_keywords,
    nr_candidates,
):
    summary = input_sentence
    from sklearn.feature_extraction.text import CountVectorizer

    n_gram_range = (number_of_words_in_each_keyword, number_of_words_in_each_keyword)
    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
        [summary]
    )
    candidates = count.get_feature_names()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    doc_embedding = model.encode([summary])
    candidate_embeddings = model.encode(candidates)

    from sklearn.metrics.pairwise import cosine_similarity

    top_n = number_of_top_keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    import numpy as np
    import itertools

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum(
            [
                distances_candidates[i][j]
                for i in combination
                for j in combination
                if i != j
            ]
        )
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def bert_keywords_diversificatin_using_maximal_marginal_relevance_method(
    input_sentence, number_of_words_in_each_keyword, number_of_top_keywords, diversity
):
    summary = input_sentence
    from sklearn.feature_extraction.text import CountVectorizer

    n_gram_range = (number_of_words_in_each_keyword, number_of_words_in_each_keyword)
    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
        [summary]
    )
    candidates = count.get_feature_names()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    doc_embedding = model.encode([summary])
    candidate_embeddings = model.encode(candidates)

    from sklearn.metrics.pairwise import cosine_similarity

    top_n = number_of_top_keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    word_embeddings = candidate_embeddings
    words = candidates

    import numpy as np

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # Calculate MMR
        mmr = (
            1 - diversity
        ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def text_summarizer(sentence_summary):
    from summarizer.sbert import SBertSummarizer

    # input_sentence = ''' Same problem with wind turbines.  They also don't think of the effect solar farms have on insects, wildlife, or the amount of land being taken that could otherwise be used for food production.  They don't think of how much vegetation is destroyed for a solar farm.  And for what?  I don't care how many times people say it is a settled question, those of us older people who have been around a while will tell you that things really are no different than they have been, and the doom-and-gloom prognostications from Al Gore and others for decades have not come true or even close to coming true.In the end, China will end up “disposing” of our energy waste products at a great price to us. Their companies will strip the panels of useable materials and dump the rest of the waste into international waters.'''

    model = SBertSummarizer("paraphrase-MiniLM-L6-v2")
    result = model(sentence_summary, num_sentences=3)
    return result


def text_similarity_2(first_sentence, second_sentence):
    from sentence_transformers import SentenceTransformer

    sentence_1 = (
        first_sentence  # "Three years later, the coffin was still full of Jello."
    )
    sentence_2 = second_sentence  # "The person box was packed with jelly many dozens of months later."
    sentences = []
    sentences.append(sentence_1)
    sentences.append(sentence_2)

    model = SentenceTransformer("bert-base-nli-mean-tokens")
    sentence_embeddings = model.encode(sentences)
    from sklearn.metrics.pairwise import cosine_similarity

    text_similarity = cosine_similarity(
        [sentence_embeddings[0]], [sentence_embeddings[1]]
    )
    # type(text_similarity)
    if float(text_similarity[0]) > 0.7:
        output_sim = "similar"
    else:
        output_sim = "not similar"
    return output_sim

    # return str(float(text_similarity[0]))


def sentiment(sentence_sentiment):
    # Importing the pipeline function from the transformers
    from transformers import pipeline

    # Creating a TextClassificationPipeline for Sentiment Analysis
    pipe = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    # Analyzing sentiment
    pipe(sentence_sentiment)
    # print(type(pipe(sentence_sentiment)))
    return pipe(sentence_sentiment)[0]["label"]


def text_summarizer(sentence_summary):
    from summarizer.sbert import SBertSummarizer

    # input_sentence = ''' Same problem with wind turbines.  They also don't think of the effect solar farms have on insects, wildlife, or the amount of land being taken that could otherwise be used for food production.  They don't think of how much vegetation is destroyed for a solar farm.  And for what?  I don't care how many times people say it is a settled question, those of us older people who have been around a while will tell you that things really are no different than they have been, and the doom-and-gloom prognostications from Al Gore and others for decades have not come true or even close to coming true.In the end, China will end up “disposing” of our energy waste products at a great price to us. Their companies will strip the panels of useable materials and dump the rest of the waste into international waters.'''
    model = SBertSummarizer("paraphrase-MiniLM-L6-v2")
    result = model(sentence_summary, num_sentences=2)
    return result


def toxicity(sentence_toxicity):
    from detoxify import Detoxify

    results = Detoxify("original").predict(sentence_toxicity)
    if results["toxicity"] > 0.4:
        output = "toxic"
    else:
        output = "not toxic"
    return output


app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/*": {"origins": "http://localhost:port"}})

# api_v1_cors_config = {
#   "origins": ["*"],
# "methods": ["OPTIONS", "GET", "POST"],
#   "allow_headers": ["Content-Type"]
# }
# CORS(app, resources={"/*": api_v1_cors_config})


# CORS(app)

# app = Flask(__name__)
# app.debug = True
# config = None

# app.config["SECRET_KEY"] = "the quick brown fox jumps over the lazy   dog"
# app.config["CORS_HEADERS"] = "Content-Type"
# # cors = CORS(app)
# # cors = CORS(app, resources={r'/alltext/*': {'origins': '*'}})

# CORS(app)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="(%(threadName)-10s) %(message)s",
# )





@app.route("/<string:name>/")
@cross_origin()
def hello(name):
    # extracted_words = extract_keywords_using_rake(summarize_bert(name))
    extracted_words = extract_keywords_using_distil_bert(name, 1, 5)
    # extracted_words = bert_keywords_diversificatin_using_max_sum_similarity_method(name, 1, 6, 2)
    extracted_words = ", ".join(extracted_words)
    print(extracted_words)
    return extracted_words


@app.route("/sentiment/<string:name1>/")
@cross_origin()
def hello1(name1):
    sentiment1 = sentiment(name1)
    sentiment_string = "{'status':'True', 'sentiment':'" + sentiment1 + "'}"
    Dict_sentiment = eval(sentiment_string)
    json_sentiment = json.dumps(Dict_sentiment, indent=4)
    # print(json_sentiment)
    return json_sentiment


# positive, negative


@app.route("/summary/<string:name2>/")
@cross_origin()
def hello2(name2):
    summary1 = text_summarizer(name2)
    summary_string = "{'status':'True', 'summary':'" + summary1 + "'}"
    Dict_summary = eval(summary_string)
    json_summary = json.dumps(Dict_summary, indent=4)
    # print(json_summary)
    return json_summary


# summary


@app.route("/similarity/<string:name3>/<string:name4>/")
@cross_origin()
def hello3(name3, name4):
    similarity1 = text_similarity_2(name3, name4)
    similarity1_string = "{'status':'True', 'similarity':'" + similarity1 + "'}"
    Dict_similarity = eval(similarity1_string)
    json_similarity = json.dumps(Dict_similarity, indent=4)
    # print(json_similarity)
    return json_similarity


# score between 0 to 1, closer to 0 means not similiar and closer to 1 means more similar
# toxicity will be yes or no. Yes means


@app.route("/toxicity/<string:name5>/")
@cross_origin()
def hello4(name5):
    toxicity1 = toxicity(name5)
    toxicity1_string = "{'status':'True', 'result':'" + toxicity1 + "'}"
    Dict_toxicity = eval(toxicity1_string)
    json_toxicity = json.dumps(Dict_toxicity, indent=4)
    # print(json_toxicity)
    # return json_similarity
    return json_toxicity


@app.route("/alltext/", methods=["POST"])
@cross_origin(origin='localhost',headers=['Content- Type'])
def query_records():
    name = request.get_json()
    my_dictionary = dict()
    print(name["dosimilarity"])
    count = 0
    # if(name["dosimilarity"]):
    #   for i in name["alltext"]:
    #     for key, value in i.items():
    #       for j in i[key]:
    #         print(j)
    #         print("after append")
    #         my_dictionary.clear()
    #         my_dictionary[j]= "Positive"#sentiment(j)
    #         my_dictionary_copy = my_dictionary.copy()
    #         j = my_dictionary_copy
    #         # print(name["alltext"][i])
    #         print(j)

    if name["dosentiment"]:
        for index in range(len(name["alltext"])):
            for key, value in name["alltext"][index].items():
                for index2 in range(len(name["alltext"][index][key])):
                    print(name["alltext"][index][key][index2])
                    print("after append")
                    my_dictionary.clear()
                    my_dictionary[name["alltext"][index][key][index2]] = sentiment(
                        name["alltext"][index][key][index2]
                    )
                    my_dictionary_copy = my_dictionary.copy()
                    name["alltext"][index][key][index2] = my_dictionary_copy
                    # print(name["alltext"][i])
                    print(name["alltext"][index][key][index2])
        return name  # jsonify({'error': 'data not found'})

    if name["dotoxicity"]:
        for index in range(len(name["alltext"])):
            for key, value in name["alltext"][index].items():
                for index2 in range(len(name["alltext"][index][key])):
                    print("sentence is ")
                    print(name["alltext"][index][key][index2])
                    print("after append")
                    my_dictionary.clear()
                    my_dictionary[name["alltext"][index][key][index2]] = toxicity(
                        name["alltext"][index][key][index2]
                    )
                    my_dictionary_copy = my_dictionary.copy()
                    name["alltext"][index][key][index2] = my_dictionary_copy
                    # print(name["alltext"][i])
                    print(name["alltext"][index][key][index2])
                    
        return name

    if name["dosimilarity"]:
        for index in range(len(name["alltext"])):
            for key, value in name["alltext"][index].items():
                for index2 in range(len(name["alltext"][index][key])):
                    print(name["alltext"][index][key][index2])
                    print("after append")
                    my_dictionary.clear()
                    my_dictionary[
                        name["alltext"][index][key][index2]
                    ] = text_similarity_2(
                        name["alltext"][index][key][index2], name["clickedtext"]
                    )
                    my_dictionary_copy = my_dictionary.copy()
                    name["alltext"][index][key][index2] = my_dictionary_copy
                    # print(name["alltext"][i])
                    print(name["alltext"][index][key][index2])
        # name.header.add("Access-Control-Allow-Origin", "*")
        # name.header.add("Access-Control-Allow-Headers", "*")
        # name.header.add("Access-Control-Allow-Methods", "*")
        return name

    my_list = []
    my_list.clear()
    if name["dosummary"]:
        for index in range(len(name["alltext"])):
            for key, value in name["alltext"][index].items():
                for index2 in range(len(name["alltext"][index][key])):
                    print(name["alltext"][index][key][index2])
                    print("after append")
                    my_list.append(name["alltext"][index][key][index2])

        my_string = ".".join(my_list)
        name["summarytext"] = text_summarizer(my_string)

        # name.header.add("Access-Control-Allow-Origin", "*")
        # name.header.add("Access-Control-Allow-Headers", "*")
        # name.header.add("Access-Control-Allow-Methods", "*")
        return name


if __name__ == "__main__":
    extracted_words = extract_keywords_using_distil_bert(
        "Lawton, who was sentenced to 12 years in prison for stealing roughly $12 million in diamonds and gold from jewelers at gunpoint, said the smash-and-grabs happening across the United States are organized and could be related to gang initiations. However, he said, they could be prevented",
        3,
        5,
    )

    app.run(host="0.0.0.0")
    # app.run()
