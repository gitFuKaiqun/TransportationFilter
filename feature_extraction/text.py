#!/usr/bin/python
# -*- coding: utf8 -*-

#from nltk.util import bigrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.fixes import Counter
import math
log2 = lambda x : math.log(x,2.0)
pow2 = lambda x : math.pow(x,2)
# ---- Refine Vocabulary -------------
class SelectionMeasures():

  def mutualinformation(self, n11,n10,n01,n00):
    n1_ = float(n10 + n11)
    n0_ = float(n01 + n00)
    n_1 = float(n01 + n11)
    n_0 = float(n10 + n00)
    n   = n1_ + n0_
    return math.fsum( [ ( (n11/n)*log2((n * n11)/(n1_ * n_1)) ),
                        ( (n01/n)*log2((n * n01)/(n0_ * n_1)) ),
                        ( (n10/n)*log2((n * n10)/(n1_ * n_0)) ),
                        ( (n00/n)*log2((n * n00)/(n0_ * n_0)) )
                      ]
                    )

  def chi_sqr(self,n11,n10,n01,n00):
    return (  (math.fsum([ n11, n10, n01, n00] ) *
               pow2( ((n11 * n00) - (n10 * n01)) )
              ) /
              ( math.fsum([n11, n01]) *
                math.fsum([n11, n10]) *
                math.fsum([n10, n00]) *
                math.fsum([n01, n00])
              )
           )

class FeatureSelection(SelectionMeasures):

  def selectfeatures(self,term_counts_per_doc=None,vocabulary=None,labels=None,
                          select_top_k=0,measure='mi',mutualExclusion=True):
    n11,n10,n01,n00 = 0.0,0.0,0.0,0.0
    labels = labels.tolist()
    uniq_labels = Counter(labels).keys()
    features = {}
    if measure == 'mi':
      measure = self.mutualinformation
    else:
      measure = self.chi_sqr
    for c in uniq_labels:
      features[c] = []
      for term in vocabulary:
        for index, doc in enumerate(term_counts_per_doc):
          if doc[term]:
            if labels[index] == c:
              n11 += 1.0
            else:
              n10 += 1.0
          else:
            if labels[index] == c:
              n01 += 1.0
            else:
              n00 += 1.0
        if n11 == 0.0 or n10 == 0.0  or n01 == 0.0 or n00 == 0.0:
          continue
        else:
          val = measure(n11,n10,n01,n00)
          features[c].append((term,val))
      features[c] = sorted(features[c], key= lambda x: x[1],reverse=True)

    _features = Counter()
    for c in uniq_labels:
      top_k = features[c][:select_top_k]
      _features.update([term for term,val in top_k])

    if mutualExclusion:
      _features = filter(lambda x: _features[x] == 1,_features)
    else:
      _features = _features.keys()

    return _features

# --- NGrams mixin ----------------
class NGramMixin:

  def _word_ngrams(self,tokens,stop_words=None):
    # Turn tokens into sequence of n-grams after stop words filtering and stemming#
    if stop_words is None:
      stop_words = []

    tokens = [self.stem(w) for w in tokens if w.encode(self.charset) not in stop_words
             and len(w) > 1 and not w.isdigit()]
    # handle token n-grams
    if self.min_n != 1 or self.max_n != 1:
      original_tokens = tokens
      tokens = []
      n_original_tokens = len(original_tokens)
      for n in xrange(self.min_n,
                      min(self.max_n + 1, n_original_tokens + 1)):
        for i in xrange(n_original_tokens - n + 1):
          tokens.append(u" ".join(original_tokens[i: i + n]))
    return tokens

from nltk.corpus import stopwords
import re
import unicodedata
def strip_accents_unicode(s):
  return u''.join([c for c in unicodedata.normalize('NFKD', s)
                    if not unicodedata.combining(c)])

class FeatureCountVectorizer(NGramMixin,CountVectorizer,FeatureSelection):

  word_pattern = re.compile(ur"\b\w\w+\b")
  stop_words_spanish = stopwords.words('spanish')
  def __init__(self,input='content', charset='utf-8',
                charset_error='strict', strip_accents=None,
                lowercase=True, preprocessor=None, tokenizer=None,
                stop_words=None, token_pattern=ur"\b\w\w+\b",
                min_n=1, max_n=1, analyzer='word',max_df=1.0,
                max_features=None,vocabulary=None, measure='mi',
                top_k = 0.40, mutualExclusion=True,
                binary=False, dtype=long,stemmer=None,min_df=0.01):

    if stemmer is not None and hasattr(stemmer,'stem'):
      self.stem = stemmer.stem
    else:
      self.stem = lambda x: x #noop

    self.min_df = min_df
    self.measure = measure
    self.top_k = top_k
    self.mutualExclusion = mutualExclusion

    super(FeatureCountVectorizer,self).__init__(
      input=input, charset=charset, charset_error=charset_error,
      strip_accents=strip_accents, lowercase=lowercase,
      preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
      stop_words=stop_words, token_pattern=token_pattern, min_n=min_n,
      max_n=max_n, max_df=max_df,max_features=max_features,
      vocabulary=vocabulary, binary=False, dtype=dtype)

  @staticmethod
  def preprocess_unicode_text(x,stem=lambda x: x):
    x = strip_accents_unicode((x.lower()))
    x = FeatureCountVectorizer.word_pattern.findall(x)
    tokens = [stem(w) for w in x if w.encode('utf-8') not in FeatureCountVectorizer.stop_words_spanish
             and len(w) > 1 and not w.isdigit()]
    return u" ".join(tokens)
  """
  def preprocess_unicode_text(x):
    x = strip_accents_unicode((x.lower()))
    x = FeatureCountVectorizer.word_pattern.findall(x)
    tokens = [w for w in x if w.encode('utf-8') not in FeatureCountVectorizer.stop_words_spanish
             and len(w) > 1 and not w.isdigit()]
    return u" ".join(tokens)
  """
  def fit_transform(self, raw_documents, y=None,labels=None):
    """Learn the vocabulary dictionary and return the count vectors

       This is more efficient than calling fit followed by transform.
       Parameters
       ----------
       raw_documents: iterable
           an iterable which yields either str, unicode or file objects

       Returns
       -------
       vectors: array, [n_samples, n_features]
    """
    if self.fixed_vocabulary:
      # No need to fit anything, directly perform the transformation.
      # We intentionally don't call the transform method to make it
      # fit_transform overridable without unwanted side effects in
      # TfidfVectorizer
      analyze = self.build_analyzer()
      term_counts_per_doc = [Counter(analyze(doc))
                             for doc in raw_documents]
      return self._term_count_dicts_to_matrix(term_counts_per_doc)

    self.vocabulary_ = {}
    # result of document conversion to term count dicts
    term_counts_per_doc = []
    term_counts = Counter()

    # term counts across entire corpus (count each term maximum once per
    # document)
    document_counts = Counter()

    max_df = self.max_df
    min_df = self.min_df
    max_features = self.max_features
    measure = self.measure
    mutualExclusion = self.mutualExclusion
    top_k = self.top_k

    analyze = self.build_analyzer()

    for doc in raw_documents:
      term_count_current = Counter(analyze(doc))
      term_counts.update(term_count_current)

      if max_df < 1.0:
        document_counts.update(term_count_current.iterkeys())

      term_counts_per_doc.append(term_count_current)

    n_doc = len(term_counts_per_doc)

    # filter out stop words: terms that occur in almost all documents
    if max_df < 1.0:
      max_document_count = max_df * n_doc
      min_document_count = min_df * n_doc
      stop_words = set(t for t, dc in document_counts.iteritems()
                         if dc > max_document_count or dc < min_document_count)
    else:
      stop_words = set()

    # list the terms that should be part of the vocabulary
    if max_features is None:
      terms = set(term_counts) - stop_words
    else:
      # extract the most frequent terms for the vocabulary
      terms = set()
      for t, tc in term_counts.most_common():
        if t not in stop_words:
          terms.add(t)
        if len(terms) >= max_features:
          break

    # store the learned stop words to make it easier to debug the value of
    # max_df
    self.max_df_stop_words_ = stop_words
    if labels is not None:
        select_top_k = int(top_k * len(terms))
        # feature selection
        terms = self.selectfeatures(term_counts_per_doc=term_counts_per_doc,
                               vocabulary=terms,labels=labels,
                               select_top_k=select_top_k,
                               measure=measure,mutualExclusion=mutualExclusion)

    # store map from term name to feature integer index: we sort the term
    # to have reproducible outcome for the vocabulary structure: otherwise
    # the mapping from feature name to indices might depend on the memory
    # layout of the machine. Furthermore sorted terms might make it
    # possible to perform binary search in the feature names array.
    self.vocabulary_ = dict(((t, i) for i, t in enumerate(sorted(terms))))

    if len(self.vocabulary_) == 0:
      return False

    # the term_counts and document_counts might be useful statistics, are
    # we really sure want we want to drop them? They take some memory but
    # can be useful for corpus introspection
    return self._term_count_dicts_to_matrix(term_counts_per_doc)

class FeatureTfidfTransformer(TfidfTransformer):

  def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
    super(FeatureTfidfTransformer,self).__init__(norm=norm,
          use_idf=use_idf,smooth_idf=smooth_idf,sublinear_tf=sublinear_tf)

