# --- Import libraries ---
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# --- Functions ---
def clean_each_phrase(phrase_as_list_of_words, return_type=None):
    """
    Process each individual phrase (called by function clean_phrases)

    Arguments
    ---------
    :param phrase_as_list_of_words:     a phrase as a list of words

    Optional Arguments
    ------------------
    :param return_type:                 type of processed tweet to return
                                        default to None
                                        can also choose ["stemmed", "lemmed"]

    Return
    ------
    :return:                            processed phrase depending on the chosen argument return_type
    """

    # convert to lower case
    tokens = [word.lower() for word in phrase_as_list_of_words]

    # no stemmed nor lemmed
    norm = " ".join(tokens)

    # PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    stemmed = " ".join(stemmed)

    # WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(word) for word in tokens]
    lemmed = " ".join(lemmed)

    # return
    if return_type == "stemmed":
        return stemmed
    elif return_type == "lemmed":
        return lemmed
    else:
        return norm


def clean_phrases(df):
    """
    Process the entire phrase data frame

    Arguments
    ---------
    :param df:      phrase data frame to process

    Return
    ------
    :return:        processed phrase data frame
    """

    # drop the PhraseId and SentenceId columns
    df.drop(["PhraseId", "SentenceId"], axis=1, inplace=True)

    # clean each individual tweet
    df["PhraseNorm"] = df.Phrase.apply(lambda phrase: clean_each_phrase(phrase.split()))
    df["PhraseStemmed"] = df.Phrase.apply(lambda phrase: clean_each_phrase(phrase.split(), "stemmed"))
    df["PhraseLemmed"] = df.Phrase.apply(lambda phrase: clean_each_phrase(phrase.split(), "lemmed"))

    # return
    return df
