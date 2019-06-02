import base64
import os
import string

import markdown as markdown
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
wordnet_lemmatizer = WordNetLemmatizer()

destination_path = '../data/data_with_readme_topics.csv'

random_state = 360

run_grid_search = False


def read_data(dir_path):
    data = []
    for file in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file)
        if os.path.isfile(full_path):
            with open(full_path, 'r') as content_file:
                content = content_file.read()
                md = base64.b64decode(content).decode('utf-8')
                data.append(md)

    return data


def convert_md_to_text(data):
    data = markdown.markdown(data)
    data = ''.join(BeautifulSoup(data, 'html.parser').findAll(text=True))

    return data


def get_topics(model, x, feature_names, no_top_words):
    topics = []
    y = model.transform(x)
    for row in y:
        topic_index = row.argmax()
        topic = model.components_[topic_index]
        topic_label = ' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append('Topic ' + str(topic_index) + ': ' + topic_label)

    return topics


def write_topics_to_csv(topics):
    data = pd.read_csv(r'../data/data.csv')
    data = data.assign(readme_topics=pd.Series(topics).values)
    csv = data.to_csv()

    with open(destination_path, 'w', encoding='utf-8') as f:
        f.write(csv)


def save_best_model(params, score, perplexity):
    with open('../data/models/lda_params.txt', 'w', encoding='utf-8') as f:
        f.write(f'Best params: {params}\n')
        f.write(f'Best score Score: {score}\n')
        f.write(f'Model Perplexity: {perplexity}\n')


def perplexity_score(estimator, data):
    estimator.fit_transform(data)
    score = estimator.perplexity(data)

    return -score


def grid_search(data_vectorized, lda):
    # Define Search Param
    search_params = {'n_components': [50], 'doc_topic_prior': np.linspace(0, 0.1, 3),
                     'topic_word_prior': np.linspace(0, 0.1, 3)}

    # Init Grid Search Class
    model = GridSearchCV(estimator=lda, scoring=perplexity_score,
                         param_grid=search_params, verbose=20, cv=5)

    # Do the Grid Search
    model.fit(data_vectorized)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print('Best Model's Params: ', model.best_params_)

    # Log Likelihood Score
    print('Best Log Likelihood Score: ', model.best_score_)

    # Perplexity
    print('Model Perplexity: ', best_lda_model.perplexity(data_vectorized))

    save_best_model(model.best_params_, model.best_score_, best_lda_model.perplexity(data_vectorized))

    return best_lda_model


def preprocess(data):
    data = convert_md_to_text(data)

    # Split into words
    tokens = word_tokenize(data)

    # Convert to lower case
    tokens = [w.lower() for w in tokens]

    # Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # Lematize
    return ' '.join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in words])


def main():
    dir_path = '../data/readme'
    print('Reading data...')
    data = read_data(dir_path)
    no_top_words = 10

    print('Preprocessing data..')
    data = list(map(preprocess, data))
    # nltk.pprint(data[:1])

    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(data))

    wordcloud.to_file('../data/graph/readme.png')

    # LDA can only use raw term counts because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(analyzer='word', max_df=0.9, min_df=10)
    tf = tf_vectorizer.fit_transform(data)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print('Number of features: ', str(len(tf_feature_names)))

    # Init the Model
    lda = LatentDirichletAllocation(random_state=random_state, n_components=50,
                                    doc_topic_prior=0.05, topic_word_prior=0.1)

    if run_grid_search:
        print('Running grid search..')
        lda = grid_search(tf, lda)

    lda = lda.fit(tf)
    print('Model Perplexity: ', lda.perplexity(tf))

    topics = get_topics(lda, tf, tf_feature_names, no_top_words)

    write_topics_to_csv(topics)


if __name__ == '__main__':
    main()
