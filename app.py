from flask import Flask, request, jsonify, render_template
import joblib
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app=Flask(__name__)
model =tf.keras.models.load_model('my_model.keras')
sc = joblib.load('standard_scaler.pkl')
cv = joblib.load('cv.pkl')

all_stopwords = stopwords.words('english')
important_stopwords = [
    'not', 'no', 'never', 'neither', 'none', 'nobody', 'nowhere', 'nothing', 'nor',
    'very', 'really', 'extremely', 'absolutely', 'quite', 'barely', 'hardly', 'scarcely',
    'but', 'however', 'although', 'though', 'yet', 'despite'
]
for i in important_stopwords:
    if i in all_stopwords:
        all_stopwords.remove(i)
all_stopwords.append("br")
@app.route('/')
def Home():
    return render_template("app.html")
@app.route("/predict",methods=["POST"])
def prediction():
    review = [x for x in request.form.values()]
    review = review[0]
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    print(review)
    x=cv.transform([review]).toarray()
    x=sc.transform(x)
    print(model.predict(x)[0][0])
    if model.predict(x)[0][0]>0.5:
        return render_template("app.html",msg="🌟 Thank you for your glowing review! We're thrilled that you enjoyed the movie as much as we did. 🎬🍿")
    else:
        return render_template("app.html",msg="😔 We're sorry to hear that the movie didn't meet your expectations.Thank you for sharing! 🙏")
if __name__=="__main__":
    app.run()