from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import re

def format_datasets(datasets_list):
    """
    Formats a list of datasets, where specific formatting rules apply to the first dataset
    in the list. For the first dataset only: 
    - It inverses binary values in the second column (1 becomes 0, and 0 becomes 1).
    - It strips quotes from the beginning and end of strings in the first column.

    Args:
        datasets_list (list of list of lists): A list where each element is a dataset 
            represented as a list of rows, and each row is a list of values.

    Returns:
        list: A single list of all rows from all datasets after applying the specified
            formatting to the first dataset in the list.
    """
    formatted_db = []
    for d, dataset in enumerate(datasets_list):
        for i, row in enumerate(dataset):  
            if d == 0:  
                if row[1] == 1:
                    row[1] = 0
                else:
                    row[1] = 1
                    
                if row[0] == '"':
                    row[0] = row[1:-1]

            formatted_db.append(row)  

    return formatted_db

def get_infos_from_list_of_sentences(list_of_sentences):
    """
    Processes a list of sentence records by adding WordNet information to each sentence, stemming the words, 
    and then compiling a list with the updated sentences and their original associated data.

    Args:
        list_of_sentences (list of list/tuple): Each inner list or tuple should contain three elements: 
            an identifier, the original sentence as a string, and additional associated data.

    Returns:
        list of list: A processed list where each element is a list containing the identifier, 
            the processed and stemmed sentence, and the original additional data.
    """
    output = []
    for index, record in enumerate(list_of_sentences):
        sentence = record[1]
        sentence = add_wordnet_info_to_token(record[1])    
        sentence = get_stems_from_sentence(sentence)
        output.append([record[0], sentence, record[2]])
    return output 
        
def get_infos_from_list_of_sentences_framenet(list_of_sentences):
    """
    Processes a list of sentence records by adding WordNet information to each sentence, stemming the words, 
    and then compiling a list with the updated sentences and their original associated data.

    Args:
        list_of_sentences (list of list/tuple): Each inner list or tuple should contain three elements: 
            an identifier, the original sentence as a string, and additional associated data.

    Returns:
        list of list: A processed list where each element is a list containing the identifier, 
            the processed and stemmed sentence, and the original additional data.
    """
    output = []
    for index, record in enumerate(list_of_sentences):
        sentence = record[1]
        sentence = add_framenet_info_to_token(record[1])    
        sentence = get_stems_from_sentence(sentence)
        output.append([record[0], sentence, record[2]])
    return output 

def get_stems_from_sentence(sentence): 
    """
    Processes the input sentence to extract the stem of each word using the Snowball stemmer.

    Args:
        sentence (str): The sentence from which stems are to be extracted.

    Returns:
        list of str: A list of stemmed words from the given sentence. Returns an empty list if the input is not a valid string or is empty.
    """
    if not isinstance(sentence, str) or sentence == "":
        return []
    stemmer = SnowballStemmer('english')
    tokens = process_tokens(word_tokenize(sentence))
    return [stemmer.stem(token) for index, token in enumerate(tokens)]

def process_tokens(tokens):
    """
    Process a list of strings removing punctuation and stop words.

    Args:
        tokens (list of str): List of tokens to clean.

    Returns:
        list of str: List of strings without punctuation and stop words.
    """
    output = []
    stop_words = stopwords.words('english')
    for index, token in enumerate(tokens):
        if token.lower() not in stop_words and token.lower() not in string.punctuation:
            output.append(token)
    return output

def add_wordnet_info_to_token(context):
    """
    Enhances a given text by appending the WordNet definition of each unique token 
    that is not a stop word or punctuation, provided a definition exists.

    Args:
        context (str): The text string to be processed.

    Returns:
        str: The original text string concatenated with the unique tokens and 
            their corresponding WordNet definitions.
    """
    if not isinstance(context, str):
        context = str(context)  
    
    tokens = []
    for token in list(set(process_tokens(word_tokenize(context)))):
        try:
            synset = lesk(token, context)
            if synset:  
                definition = synset.definition()
                tokens.append(token)
                tokens.append(definition)
        except TypeError as e:
            print(f"Error processing token '{token}': {e}")
    
    tokens = list(set(tokens))
    return context + " " + " ".join(tokens)

def overlap_context_synset(context, synset):
    """
    Calculates the number of words in common between a given context (a set of words) and 
    the WordNet synset's definition, its examples, and the definitions and examples of its hypernyms.

    Args:
        context (set of str): A set of words representing the context against which to compare.
        synset (Synset): A WordNet synset object from the NLTK library.

    Returns:
        int: The number of unique words that occur both in the context and in the combined 
            glosses (definitions and examples) of the synset and its hypernyms.
    """
    gloss = set(word_tokenize(synset.definition()))
    for example in synset.examples():
        gloss.union(set(word_tokenize(example)))
    for hyper in synset.hypernyms():
        gloss.union(set(word_tokenize(hyper.definition())))
        for example in hyper.examples():
            gloss.union(set(word_tokenize(example)))
    
    return len(gloss.intersection(context))

def lesk(token, context):
    """
    Implements the simplified Lesk algorithm to determine the most likely sense of a word
    given a context. The algorithm selects the sense that has the highest overlap of words
    between the synset's definitions (and examples) and the context.

    Args:
        token (str): The word for which the sense needs to be determined.
        context (str): The text context in which the word appears, used to derive context for disambiguation.

    Returns:
        Synset: The WordNet synset that best matches the context, or None if no synset is found.
    """
    best_sense = None
    max_overlap = 0
    ctx = set(word_tokenize(context))
    synsets = wn.synsets(token)
    
    for synset in synsets:
        overlap = overlap_context_synset(ctx, synset)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset
            
    return best_sense


def get_frequent_words(records, percentage_to_mantain = 0.1):
    """
    Computes the most frequent words from a list of records and returns a dictionary 
    of these words maintained up to a specified percentage of the top occurrences.

    Args:
        records (list of lists): A list where each element is another list containing
                                identifiers and words, with words being at the second index.
        percentage_to_mantain (float, optional): The fraction of the top frequent words to retain.
                                                Defaults to 0.1 (i.e., top 10%).

    Returns:
        dict: A dictionary of words as keys and their frequencies as values, truncated to
            include only the top percentage specified by `percentage_to_mantain`.
    """
    counter = {}
    for index, record in enumerate(records):
        for index2, word in enumerate(record[1]):
            try:
                counter[word] += 1
            except:
                counter.setdefault(word, 1)
        
    counter = sorted(counter.items(), key=lambda item: item[1], reverse = True)
    percentage_to_mantain = int(len(counter) * percentage_to_mantain)
    counter = counter[:percentage_to_mantain]
    return dict(counter)

def add_framenet_info_to_token(context):
    """
    Enhances a given text by appending the FrameNet definition of each unique token 
    that is not a stop word or punctuation, provided a definition exists.

    Args:
        context (str): The text string to be processed.

    Returns:
        str: The original text string concatenated with the unique tokens and 
            their corresponding FrameNet definitions.
    """
    
    # nltk.download('framenet_v17')
    if not isinstance(context, str):
        context = str(context)  
    
    tokens = []
    for token in list(set(process_tokens(word_tokenize(context)))):
        try:
            # Removes label like "frame:", "Definition:"
            frames = fn.frames(r'.*\b{}\b'.format(re.escape(token)))

            if frames:
                frame = frames[0]
                tokens.append(f"{token} ({frame.name})")
                tokens.append(frame.definition)
        except TypeError as e:
            print(f"Error processing token '{token}': {e}")
    
    tokens = list(set(tokens))
    return context + " " + " ".join(tokens)