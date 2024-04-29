from textblob import TextBlob
import utils as ut


def sentiment_analysis(sentence):
    """
    Analyzes the sentiment of a given sentence using TextBlob and determines its polarity as positive, neutral, or negative.

    Args:
        sentence (str): The sentence whose sentiment is to be analyzed.

    Returns:
        int: A value indicating the sentiment of the sentence: 0 for neutral, -1 for negative, and 1 for positive.
    """
    analysis = TextBlob(sentence)

    if -0.05 <= analysis.sentiment.polarity <= 0.05:
        return 0
    elif analysis.sentiment.polarity < -0.05:
        return ut.get_sign(-1)
    else:
        return ut.get_sign(+1)

    

