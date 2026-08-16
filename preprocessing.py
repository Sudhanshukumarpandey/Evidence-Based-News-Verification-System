import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Why this block exists:
# NLTK requires resources to be downloaded before using them.
# We download 'stopwords', 'wordnet', and 'omw-1.4' programmatically so that the user's script
# doesn't crash on execution. We use a try-except block to check if they are already downloaded first.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize the WordNetLemmatizer.
# Lemmatization is the process of reducing a word to its root dictionary form.
# For example: "running" -> "run", "better" -> "good", "leaves" -> "leaf".
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
# Add 'reuters' to the stopwords set to prevent the models from overfitting
# to the specific news agency source label present in the real news files.
stop_words.add('reuters')

def clean_text(text):
    """
    Cleans the input text by applying basic text preprocessing techniques.
    
    Parameters:
    text (str): The raw text of a news article.
    
    Returns:
    str: The preprocessed and cleaned text.
    """
    # 1. Handle Null or non-string inputs
    if not isinstance(text, str):
        return ""
    
    # 2. Lowercase the text
    # Why: Machine learning models treat 'Police' and 'police' as different tokens. 
    # Converting to lowercase helps consolidate word counts.
    text = text.lower()
    
    # 3. Remove HTML tags
    # Why: Articles scraped from web pages might contain raw HTML like "<p>" or "<a>".
    # Syntax: '<.*?>' matches any text inside angle brackets.
    # re.sub replaces any matches of this pattern with an empty string.
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove URLs (Web links)
    # Why: Links (e.g. http://example.com) are unique and add noise rather than semantic value.
    # Syntax: re.sub matches standard URL patterns and removes them.
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 5. Remove numbers
    # Why: Numbers like dates or quantities (e.g. 100, 2026) are usually context-specific and 
    # do not help distinguish fake news from real news generalistically.
    # Syntax: '\d+' matches one or more digits.
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove punctuation marks
    # Why: Punctuation (e.g. !, ?, @, #) can split words incorrectly or be treated as distinct.
    # Syntax: We create a character class using string.punctuation and substitute matches with space.
    # string.punctuation is a built-in Python string: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punctuation_pattern = f"[{re.escape(string.punctuation)}]"
    text = re.sub(punctuation_pattern, ' ', text)
    
    # 7. Split into words (Tokenization), Remove Stopwords and Lemmatize
    # Syntax: text.split() splits the string by any whitespace (spaces, tabs, newlines).
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Check if the word is NOT in the stopwords list
        if word not in stop_words:
            # Reduce word to its lemma (base form)
            lemma_word = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma_word)
            
    # 8. Re-combine list of words back into a single string
    # Why: TF-IDF vectorizer accepts a single string block per document as input.
    # Syntax: ' '.join() takes list elements and concatenates them with a single space separator.
    text = ' '.join(cleaned_words)
    
    # 9. Remove extra whitespaces
    # Why: Double spaces or newlines might be left over from removing punctuation.
    # Syntax: re.sub(r'\s+', ' ', text) replaces multiple spaces with a single space.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
