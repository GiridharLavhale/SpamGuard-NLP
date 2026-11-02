"""
Text preprocessing and NLP feature extraction module for SpamGuard NLP.
Provides consistent text normalization, tokenization, stopword removal with negation preservation,
and heuristic indicator extraction.
"""

import re
import string
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize stemmer
_stemmer = PorterStemmer()

# Words to preserve during stopword removal (crucial for sentiment & message context)
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing',
    'nowhere', 'hardly', 'scarcely', 'barely', "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "mustn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
}

# Known spam trigger keywords and heuristic patterns
SPAM_TRIGGER_WORDS = [
    'free', 'winner', 'win', 'won', 'cash', 'prize', 'claim', 'urgent',
    'congratulations', 'congrats', 'reward', 'selected', 'guaranteed',
    'call now', 'txt', 'text', 'sms', 'ringtone', 'credit', 'loan',
    'offer', 'exclusive', 'apply', 'entry', 'valid', 'expire', 'expires',
    'urgent', 'action required', 'account', 'verify', 'bank', 'limited time',
    'click', 'link', 'subscribe', 'unsubscribe', 'stop', 'bonus', 'million',
    'thousand', 'dollars', 'pounds', 'reply'
]

# Regular expressions for entity patterns
URL_REGEX = re.compile(
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'https?:\/\/[^\s]+|bit\.ly\/[^\s]+|tinyurl\.com\/[^\s]+)',
    re.IGNORECASE
)

EMAIL_REGEX = re.compile(
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    re.IGNORECASE
)

PHONE_REGEX = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{5,7}\b|\b\d{10,12}\b'
)

CURRENCY_REGEX = re.compile(r'[\$£€¥₹]')

REPEATED_CHARS_REGEX = re.compile(r'(.)\1{2,}')


def get_stop_words() -> set:
    """Retrieve filtered stopwords set preserving negation words."""
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except Exception:
        # Fallback common stopwords if NLTK corpus isn't accessible
        nltk_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
    return nltk_stopwords - NEGATION_WORDS


STOP_WORDS = get_stop_words()


def preprocess_text(text: str, apply_stemming: bool = True) -> str:
    """
    Standardize, clean, tokenize, remove stopwords, and optionally stem text.
    Consistent between training and prediction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Normalize URLs
    cleaned = URL_REGEX.sub(' url_token ', cleaned)

    # 3. Normalize Emails
    cleaned = EMAIL_REGEX.sub(' email_token ', cleaned)

    # 4. Normalize Currency symbols to currency_token
    cleaned = CURRENCY_REGEX.sub(' currency_token ', cleaned)

    # 5. Normalize Phone numbers / shortcodes
    cleaned = PHONE_REGEX.sub(' phone_token ', cleaned)

    # 6. Compress repeated characters (e.g. 'coooool' -> 'cool')
    cleaned = REPEATED_CHARS_REGEX.sub(r'\1\1', cleaned)

    # 7. Normalize punctuation (replace with spaces except underscores in tokens)
    # We allow tokens like url_token, email_token, phone_token, currency_token
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)

    # 8. Tokenize text using regex word boundary
    tokens = re.findall(r'\b[a-z_][a-z0-9_]*\b', cleaned)

    # 9. Stopword removal and filtering
    filtered_tokens = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        if len(token) <= 1 and token not in {'no', 'ok'}:
            continue
        if apply_stemming and not token.endswith('_token'):
            token = _stemmer.stem(token)
        filtered_tokens.append(token)

    return " ".join(filtered_tokens)


def extract_heuristic_indicators(raw_text: str) -> Dict[str, Any]:
    """
    Extract transparent heuristic indicators and statistics from a raw SMS message.
    Used for UI explainability and suspicious pattern analysis.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            'character_count': 0,
            'word_count': 0,
            'uppercase_count': 0,
            'uppercase_ratio': 0.0,
            'digits_count': 0,
            'has_url': False,
            'has_email': False,
            'has_phone': False,
            'has_currency': False,
            'matched_keywords': [],
            'suspicion_score': 0.0,
            'risk_level': 'Low'
        }

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)
    uppercase_chars = sum(1 for c in raw_text if c.isupper())
    digits_chars = sum(1 for c in raw_text if c.isdigit())
    uppercase_ratio = (uppercase_chars / char_count) if char_count > 0 else 0.0

    has_url = bool(URL_REGEX.search(raw_text))
    has_email = bool(EMAIL_REGEX.search(raw_text))
    has_phone = bool(PHONE_REGEX.search(raw_text))
    has_currency = bool(CURRENCY_REGEX.search(raw_text))

    lower_text = raw_text.lower()
    matched_keywords = []
    for kw in SPAM_TRIGGER_WORDS:
        # Check whole word match or phrase match
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, lower_text):
            matched_keywords.append(kw)

    # Compute a transparent heuristic suspicion score (0 to 100)
    score = 0
    score += min(len(matched_keywords) * 15, 45)
    if has_url:
        score += 20
    if has_phone:
        score += 15
    if has_currency:
        score += 15
    if uppercase_ratio > 0.3:
        score += 15
    if digits_chars > 5:
        score += 10

    suspicion_score = min(score, 100)

    if suspicion_score >= 60:
        risk_level = "High"
    elif suspicion_score >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        'character_count': char_count,
        'word_count': word_count,
        'uppercase_count': uppercase_chars,
        'uppercase_ratio': round(uppercase_ratio * 100, 1),
        'digits_count': digits_chars,
        'has_url': has_url,
        'has_email': has_email,
        'has_phone': has_phone,
        'has_currency': has_currency,
        'matched_keywords': sorted(list(set(matched_keywords))),
        'suspicion_score': suspicion_score,
        'risk_level': risk_level
    }
