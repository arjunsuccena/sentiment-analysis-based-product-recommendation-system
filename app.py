from flask import Flask, jsonify,  request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='templates')
recommendation_model = joblib.load("./models/recommendation_model.pkl")
sentiment_analysis_model = joblib.load("./models/lr_model.pkl")

reviews = pd.read_csv('./data/sample30.csv')
reviews = reviews.assign(productId=(reviews['name']).astype('category').cat.codes)
all_users = reviews[['reviews_username']]
all_users = all_users.drop_duplicates()


def get_products():
    products = pd.DataFrame(data=reviews[['productId', 'name']])
    products.drop_duplicates(inplace=True)
    return products


def get_recommendation(username):
    recommendation20 = recommendation_model.loc[username].sort_values(ascending=False)[0:20]
    products = get_products()
    recommendation20 = products[products.productId.isin(recommendation20.index)]
    recommendation20_df = pd.DataFrame({'productId': recommendation20.productId, 'name': recommendation20.name})
    return recommendation20_df


def get_top5(recommendation20_df):
    recommendation20reviews = reviews[reviews.productId.isin(recommendation20_df.productId)]
    data = recommendation20reviews['reviews_text']
    train_data = recommendation20reviews['reviews_text']
    vectorize_word = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word',
                                     token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1, 1),
                                     max_features=10000)
    train_features_word = vectorize_word.fit_transform(data)
    vectorize_char = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char', stop_words='english',
                                     ngram_range=(2, 6), max_features=60000)
    train_features_char = vectorize_char.fit_transform(data)
    train_features = np.hstack([train_features_char, train_features_word])
    result = sentiment_analysis_model.predict(train_features_char)
    recommendation20_df = recommendation20_df.join(pd.DataFrame({'sentiment': result}))
    return recommendation20_df[recommendation20_df['sentiment'] == 'Happy']['name'][0:5]


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        m_name = request.form['username'].title().lower()
        try:
            result_final = get_recommendation(m_name)
            names = []
            productIds = []
            for i in range(len(result_final)):
                productIds.append(result_final.iloc[i][0])
                names.append(result_final.iloc[i][1])
            return render_template('positive.html', productIds=productIds, names=names, search_name=m_name)
        except:
            return render_template('negative.html', name=m_name)


if __name__ == '__main__':
    app.debug = True
    app.run()
