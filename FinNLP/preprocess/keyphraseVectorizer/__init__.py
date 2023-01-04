# Linguistic-based Keyphrase Vectorizer
# ------------------------------
# -@latest 08/29/2022
# -@inception 08/29/2022
#
# Set of vectorizers that extract keyphrases with part-of-speech patterns 
# from a collection of text documents and convert them into a document-keyphrase matrix.
#
from .keyphrase_count_vectorizer import KeyphraseCountVectorizer
from .keyphrase_tfidf_vectorizer import KeyphraseTfidfVectorizer
from .keyphrase_vectorizer_mixin import _KeyphraseVectorizerMixin
