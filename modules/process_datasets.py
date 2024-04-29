from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn


def format_datasets(datasets_list):
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
    output = []
    for index, record in enumerate(list_of_sentences):
        sentence = record[1]
        sentence = add_wordnet_info_to_token(record[1])
        print(sentence)
        break
        
        
        
        sentence = get_stems_from_sentence(sentence)
        output.append([record[0], sentence, record[2]])
    return output 
        
def get_stems_from_sentence(sentence): 
    if not isinstance(sentence, str) or sentence == "":
        return []
    stemmer = SnowballStemmer('english')
    tokens = process_tokens(word_tokenize(sentence))
    return [stemmer.stem(token) for index, token in enumerate(tokens)]

def process_tokens(tokens):
    output = []
    stop_words = stopwords.words('english')
    for index, token in enumerate(tokens):
        if token.lower() not in stop_words and token.lower() not in string.punctuation:
            output.append(token)
    return output

# Given a token a the context (sentence) of the token the function returns a string of tokens with the definition for that token
def add_wordnet_info_to_token(context):
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
 
# Returns the number of words in common between ctx and synset with his hyperonyms
def overlap_context_synset(context, synset):
    gloss = set(word_tokenize(synset.definition()))
    for example in synset.examples():
        gloss.union(set(word_tokenize(example)))
    for hyper in synset.hypernyms():
        gloss.union(set(word_tokenize(hyper.definition())))
        for example in hyper.examples():
            gloss.union(set(word_tokenize(example)))
    
    return len(gloss.intersection(context))

# Returns the most probable synset based on word and context
def lesk(token, context):
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