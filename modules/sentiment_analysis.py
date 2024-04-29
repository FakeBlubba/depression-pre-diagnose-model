from textblob import TextBlob
import utils as ut


def sentiment_analysis(sentence):
    analysis = TextBlob(sentence)

    if -0.05 <= analysis.sentiment.polarity <= 0.05:
        return 0
    elif analysis.sentiment.polarity < -0.05:
        return ut.get_sign(-1)
    else:
        return ut.get_sign(+1)

    

