# Language Modeling & Smoothing

Models that assign probabilities to sequences of words are called **language models** or **LMs**, e.g., n-gram, neural language models based on the RNNs and transformers. 

Application examples are speech recognition, spelling correction, collocation error correction, machine translation, question-answering and etc.

A language model computes the probability of a sequence of words.

## N-Grams

N-gram is a sequence of $n$ words: 2-gram is a two-word sequence of words.

**Chain rule of probability**

$$
\begin{equation*}\begin{align*}
P(X_1\ldots X_n)&=P(X_1)P(X_2|X_1)P(X_3|X_{1:2})\ldots P(X_n|X_{1:n-1})\\
&=\prod_{k=1}^n P(X_k|X_{1:k-1})\end{align*}\end{equation*}
$$

and 

$$
\begin{equation*}\begin{align*}
P(w_{1:n})&=P(w_1)P(w_2|w_1)P(w_3|w_{1:2})\ldots P(w_n|w_{1:n-1})\\
&=\prod_{k=1}^n P(w_k|w_{1:k-1})\end{align*}\end{equation*}
$$

The chain rule shows the link between computing the joint probability of a sequence and computing the conditional probability of a word given previous words.

N-gram model is instead of computing the probability of a word given its entire history, use the approximate history of just the last few words.

The bigram model approximates the probability of a word given all the previous words $P(w_n|w_{1:n-1})$ by using only the conditional probability of the preceding word $P(w_n|w_{n-1})$.

The assumption that the probability of a word depends only on the previous word is called a `Markov` assumption.

The general equation for this n-gram approximation to the conditional probability of the next word in a sequence, $N$ is the size of n-gram,

$$
P(w_n|w_{1:n-1})\approx P(w_n|w_{n-N+1:n-1})
$$

**Unigram Model**

zero-order `Markov` assumption

$$
P(x_1,x_2,\ldots,x_l) = \prod^l_{i=1}P(x_i)
$$

**Bigram Model**

First-order `Markov` assumption

$$
P(x_1,x_2,\ldots,x_l)=P(x_1)\prod^l_{i=2}P(x_i|x_{i-1})
$$

Maximum likelihood estimation:

Use **maximum likelihood estimation** or MLE to estimate probabilities. 

$$
P(x_i|x_{i-1})=\frac{count(x_{i-1},x_i)}{\sum_xcount(x_{i-1},x)}=\frac{count(x_{i-1},x_i)}{count(x_{i-1})}
$$

where special symbols `<s>` and `</s>` are added at the beginning/end of a sentence in the corpus.

The MLE estimate for the parameters of an n-gram model by getting counts from a corpus, and normalising the counts so that they lie between 0 and 1.

For general case of MLE n-gram parameter estimation:

$$
P(x_i|x_{i-N+1:n-1})=\frac{count(x_{i-N+1:i-1},x_i)}{count(x_{i-N+1:n-1})}
$$

**Trigram Model**

Second-order `Markov` assumption

$$
P(x_1,x_2,\ldots,x_l)=P(x_1)P(x_2|x_1)\prod^l_{i=3}P(x_i|x_{i-2},x_{i-1})
$$

Given the bigram assumption for the probability of an individual word, the probability of a complete word sequence is

$$
P(w_{1:n})\approx\prod_{k=1}^nP(w_k|w_{k-1})
$$

## Evaluating Language Models

**Extrinsic evaluation:** Embed the LM in an application and measure how much the application improves. It’s an *end-to-end* evaluation. Best way to evaluate.

**Intrinsic evaluation:** measures the quality of a model independent of any application. This need a test set for measuring the quality of an n-gram model.

### Perplexity

 In practice, we don’t use raw probability as our metric for evaluating language models, but a variant called **perplexity**. 

The perplexity (PPL) of a language model *on a test set* is the inverse probability of the test set, normalised by the number of words. For a test set $W=w_1w_2\ldots w_N$: ($N$ is the size of the test set)

$$
\begin{equation*}\begin{align*}
\text{perplexity}(W)&=P(w_1w_2\ldots w_N)^{-\frac{1}{N}}\\&=\sqrt[N]{\frac{1}{P(w_1w_2\ldots w_N)}}
\end{align*}\end{equation*}
$$

use the chain rule to expand the probability of $W$:

$$
\text{perplexity}(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{P(w_i|w-1\ldots w_{i-1})}}
$$

The perplexity of a test set $W$ depends on which language model we use.

- the perplexity with a unigram LM
    
    $$
    \text{perplexity}(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{P(w_i)}}
    $$
    
- the perplexity with a bigram LM
    
    $$
    \text{perplexity}(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{P(w_i|w_{i-1})}}
    $$
    

<aside>
💡 The higher the probability, the lower the perplexity, and makes a better predictor of the words in the test set.

</aside>

Note that in computing perplexities, the n-gram model $P$ must be constructed without any knowledge of the test set or any prior knowledge of the vocabulary of the test set.

The perplexity of two language models is only comparable if they use identical vocabularies.

## Smoothing

To keep a language model from assigning zero probability to these unseen events (words in vocabulary but not used in the training), need to shave off a bit of probability mass from some more frequent events and give it to the events not seen, so the result won’t overfit the training corpus. This modification is called **smoothing** or **discounting**.

Smoothing adjusts low probabilities upwards, and high probabilities downwards.

### Laplace Smoothing or add-k smoothing

The simplest way to do smoothing is to add one to all the n-gram counts, before normalising them into probabilities.

**Laplace smoothing to unigram probabilities**

The unsmoothed MLE of the unigram probability of the word $w_i$ is its count $c_i$ normalised by the total number of word tokens $N$:

$$
P(w_i) = \frac{c_i}{N}
$$

Laplace smoothing merely adds one to each count, and adjust the denominator to add extra $V$ for there are $V$ words in the vocabulary

$$
P(w_i)=\frac{c_i+1}{N+V}
$$

**Adjusted count $c^*$**

With adjusted count $c^*$, it’s easier to compare directly with the MLE counts and can be turned into a probability like an MLE count by normalising by $N$,

$$
c_i^*=(c_i+1)\frac{N}{N+V}
$$

To describe a smoothing algorithm in terms of a relative **discount $d_c$**, the ratio of the discounted counts to the original counts:

$$
d_c=\frac{c^*}{c}
$$

### Backoff and Interpolation

When computing $P(w_n|w_{n-2}w_{n-1})$ with no examples of a particular trigram $w_{n-2}w_{n-1}w_n$, estimate its probability by using the bigram probability $P(w_n|w_{n-1})$ instead. Sometimes, using **less context** is a good thing.

- **Backoff**: is to lower-order n-gram if no evidence for a higher-order n-gram, use the trigram if the evidence is sufficient, otherwise the bigram, then the unigram.
- **Interpolation**: mix the probability estimates from all the n-gram estimators, weighting and combining the trigram, bigram, and unigram counts.

In **simple linear interpolation**, combine different order n-grams by linearly interpolating them. Thus, estimate the trigram probability $P(w_n|w_{n-2}w_{n-1})$ by mixing together the unigram, bigram, and trigram probabilities, each weighted by a $\lambda$:

$$
\begin{equation*}\begin{align*}
\hat{P}(w_n|w_{n-2}w_{n-1})=&\lambda_1P(w_n)\\&+\lambda_2P(w_n|w_{n-1})\\&+\lambda_3P(w_n|w_{n-2}w_{n-1})
\end{align*}\end{equation*}
$$

The $\lambda$s must sum to 1, making the equations equivalent to a weighted average:

$$
\sum_i\lambda_i=1
$$

A more sophisticated version of linear interpolation, each $\lambda$  weight is computed by conditioning on the context. So that if there are particularly accurate counts for a particular bigram, assume the counts of the trigram based on this bigram is more trustworthy.

$$
\begin{equation*}\begin{align*}
\hat{P}(w_n|w_{n-2}w_{n-1})=&\lambda_1(w_{n-2:n-1})P(w_n)\\&+\lambda_2(w_{n-2:n-1})P(w_n|w_{n-1})\\&+\lambda_3(w_{n-2:n-1})P(w_n|w_{n-2}w_{n-1})
\end{align*}\end{equation*}
$$

**Set $\lambda$s**

The $\lambda$s are learnt from a held-out corpus, which is an additional training corpus, since it’s hold out from the training data, that are used to set hyperparameters like these $\lambda$ values.

![[./held_out.png]]

One way to find this optimal set of $\lambda$s is to use the **EM** algorithm.

One crude approach is

$$
\lambda_3=\frac{count(x_{i-2},x_{i-1})}{count(x_{i-2},x_{i-1})+\gamma},\ \lambda_2=(1-\lambda_3)\frac{count(x_{i-1})}{count(x_{i-1})+\gamma}\\\lambda_1=1-\lambda_2-\lambda_3
$$

- ensures $\lambda_i$ is larger when count is larger
- different $\lambda$s for each n-gram
- only one parameter to estimate

In a **backoff** n-gram model, if the n-gram has zero counts, approximate it by backing off to the (n-1)-gram. And continue backing off until reaching a history that has some counts.

## `Kneser-Ney` Smoothing

The `Kneser-Ney` smoothing is a popular advanced n-gram smoothing method and has its roots in a method called **absolute discounting**.

### **Absolute Discounting**

**Discounting** of the counts for frequent n-grams is necessary to save some probability mass for smoothing algorithm to distribute to the unseen n-grams.

To discount the count (e.g 4) of a n-gram by some amount, the idea is to look at a held-out corpus and just see what the count is for all those bigrams that had count (e.g. 4) in the training set.

**Absolute discounting** formalises the intuition by subtracting a fixed (absolute) discount $d$ from each count. This small discount won’t affect the result much.

- Aims to deal with sequences that occur infrequently
- The discount $d$ reserves some probability mass for the unseen n-gram
    
    $$
    P_{\text{AbsDiscount}}(x_i|x_{i-1})=\frac{\max(count(x_{i-1,x_i})-d,0)}{count(x_{i-1})}+\lambda(x_{i-1})P(x_i)\\\lambda(x_{i-1})=\frac{d}{count(x_{i-1})}|\{x:count(x_{i-1},x)>0\}|
    $$
    
- Estimate using a held-out corpus, $d=0.75$ is used

### `Kneser-Ney` Discounting

It augments absolute discounting with a more sophisticated way to handle the **lower-order unigram** distribution. 

Create a unigram model that might call $P_{\text{continuation}}$, which answers the question “how likely is $w$ to appear as a novel continuation?” instead of answering “how likely is $w$?”.

The `Kneser-ney` is to base the estimation of $P_{\text{continuation}}$ on the **number of different contexts word $w$ has appeared in**, that is the number of bigram types it completes. Every bigram type was a novel continuation the first time it was seen. Hypothesising that words that have appeared in more contexts in the past are more likely to appear in some new context as well.

The number of times a word $w$ appears as a novel continuation can be expressed as

$$
P_{\text{continuation}}(w)\propto|\{v:C(cw)>0\}|
$$

To turn this count into a probability, normalise by the total number of word bigram types,

$$
P_{\text{continuation}}(w)=\frac{|\{v:C(cw)>0\}|}{|\{(u',w'):C(u'w')>0\}|}=\frac{|\{v:C(cw)>0\}|}{\sum_{w'}|\{v:C(vw')\}>0|}
$$

A frequent word occurring in only one context will have a low continuation probability.

The final equation for Interpolated `Kneser-ney` smoothing for bigram is:

$$
P_{KN}(w_i|w_{i-1})=\frac{max(C(w_{i-1}w_i)-d,0)}{C(w_{i-1})}+\lambda(w_{i-1})P_{\text{continuation}}(w_i)
$$