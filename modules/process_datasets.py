from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import re
import manage_datasets as md
from nltk import pos_tag
from collections import deque


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

def lesk_on_multiple_words(important_words, sentence):    
    best_synset = None
    max_overlap = 0
    
    for word in important_words:
        synset = lesk(word, sentence)
        if synset:
            overlap = overlap_context_synset(set(word_tokenize(sentence)), synset)
            if overlap > max_overlap:
                max_overlap = overlap
                best_synset = synset
    
    return best_synset


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

def get_main_word_synset(sentence):
    freq_words = md.read_frequent_words()
    tokens = [word for word in word_tokenize(sentence) if word.lower() not in stopwords.words('english')]
    if not tokens:
        return False

    tagged_tokens = pos_tag(tokens)
    
    important_words = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('VB')]

    if not important_words:
        important_words = [word for word in important_words if word in freq_words]
    if not important_words:
        important_words = tokens
    synset = lesk_on_multiple_words(important_words, sentence)
    return synset


def extract_main_synsets(cases, n):
    """
    Extracts the main synsets from the given cases, selecting the top n synsets 
    associated with the highest frequency in the sentences.

    Args:
        cases (list of tuples): A list of cases where each case is a tuple containing 
                                an identifier and a sentence.
        n (int): The number of main synsets to extract.

    Returns:
        list: A sorted list 
    """
    temp = {}
    limit = 0

    for index, sentence in enumerate(cases):
        sentence = sentence[1]
        if limit == n:
            output = sorted(temp, key=temp.get, reverse=True)[:n]
            return output
        elif get_main_word_synset(sentence):
            synset = get_main_word_synset(sentence)
            if synset in temp.keys():
                temp[synset] = 0
                limit =+ 1
            temp[synset] =+ 1
    output = sorted(temp, key=temp.get, reverse=True)[:n]
    return output


def expand_synset(synset, levels):
    """
    Expand a synset to its hypernyms for a given number of levels.

    Args:
        synset (wn.Synset): The synset to expand.
        levels (int): The number of levels to expand.

    Returns:
        list: A list of expanded synsets.
    """
    expanded = set([synset])
    current_level = set([synset])
    for _ in range(levels):
        next_level = set()
        for syn in current_level:
            next_level.update(syn.hypernyms())
        current_level = next_level
        expanded.update(current_level)
    return list(expanded)

def find_max_similarity(synsets, target_synset, levels):
    """
    Find the maximum similarity between a target synset and a list of synsets,
    expanding each synset to its hypernyms for a given number of levels.

    Args:
        synsets (list): The list of synsets to compare.
        target_synset (wn.Synset): The target synset.
        levels (int): The number of levels to expand.

    Returns:
        float: The maximum similarity score found.
        str: The synset with the maximum similarity score.
    """
    max_similarity = 0.0
    best_match = None
    expanded_target_synsets = expand_synset(target_synset, levels)

    for synset_str in synsets:
        synset = wn.synset(synset_str)
        expanded_synsets = expand_synset(synset, levels)
        for expanded_synset in expanded_synsets:
            for expanded_target_synset in expanded_target_synsets:
                similarity = synset_similarity(expanded_synset, expanded_target_synset)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
                    best_match = synset_str

    return max_similarity, best_match


def synset_similarity(synset1, synset2):
    """
    Calculate the similarity between two synsets using the Leacock-Chodorow similarity.

    Args:
        synset1 (wn.Synset): The first synset.
        synset2 (wn.Synset): The second synset.

    Returns:
        float: The similarity score between the two synsets.
    """
    if isinstance(synset1, str):
        synset1 = wn.synset(synset1)
    if isinstance(synset2, str):
        synset2 = wn.synset(synset2)
    return synset1.lch_similarity(synset2)

def find_synset_level(synsets, target_synset_name, threshold, max_levels):
    """
    Traverse the list of synsets and return the maximum similarity and the corresponding synset
    if the similarity exceeds the threshold, otherwise return -1 and the maximum similarity.

    Args:
        synsets (list): The list of synsets to search.
        target_synset_name (str): The target synset to search for.
        threshold (float): The similarity threshold to consider a synset a match.
        max_levels (int): The maximum number of levels to explore.

    Returns:
        Tuple[int, float]: A tuple containing the level of the tree where the target synset was found (or -1 if not found) and the maximum similarity score obtained.
    """
    target_synset = wn.synset(target_synset_name)
    queue = deque([(synsets, 0)])  # (current_synsets, current_level)
    max_similarity = 0.0
    best_match = None

    while queue:
        current_synsets, level = queue.popleft()
        if level > max_levels:
            break
        similarities = []

        for synset_str in current_synsets:
            try:
                if isinstance(synset_str, str):
                    synset = wn.synset(synset_str)
                else:
                    synset = synset_str
                
                similarity = synset_similarity(synset, target_synset)
                similarities.append((similarity, synset))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = synset
            except Exception as e:
                continue

        if similarities:
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_similarities = similarities[:max(1, len(similarities) * 1 // 2)]  

            for similarity, synset in top_similarities:
                if similarity >= threshold:
                    print(f"Level: {level}, Similarity: {similarity}, Synset: {synset.name()}")
                    return (level, similarity)

            next_level_synsets = []
            for _, synset in top_similarities:
                next_level_synsets.extend(synset.hypernyms())

            queue.append((next_level_synsets, level + 1))

    print(f"Max similarity for '{target_synset_name}': {max_similarity} - {best_match.name() if best_match else 'None'}")
    return (-1, max_similarity)

