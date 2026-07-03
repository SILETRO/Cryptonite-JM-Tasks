# Concept Questions and Error Analysis

### Phase I: N-Grams
1. **Why does perplexity decrease as context size increases?**
   As context size increases, the model has more information to predict the next token, which makes its predictions more certain. Higher average probability lowers the perplexity.
2. **Why do N-gram models eventually become impractical?**
   The number of possible N-grams grows exponentially with $N$. This leads to severe data sparsity and massive memory requirements.
3. **What assumptions are made by a Bigram Model?**
   The Markov assumption: the probability of a word depends only on the immediately preceding word.

### Phase II: Subword Tokenization
1. **Why do GPT models use subword tokenization?**
   It balances the small vocabulary size of character-level models with the meaningful representations of word-level models. It is not practical to include every word of dictionary. It also helps in handling new unknown words.
2. **How does BPE help with rare words?**
   Rare words are broken down into more frequent subword pieces. This prevents Out-Of-Vocabulary (OOV) errors because even an unknown word can be represented by a sequence of known subwords.
3. **Why are word-level vocabularies problematic?**
   They require massive vocabulary sizes to capture all variations of words and cannot handle completely new words without assigning an `<UNK>` token.

### Phase III: Positional Information
1. **Why does attention require positional information?**
   Self-attention operations are permutation-invariant; without positional encodings, the model treats the input as a bag of words and cannot distinguish between "dog bites man" and "man bites dog".
2. **What advantages do learned embeddings provide?**
   They allow the model to learn task-specific positional representations directly from the data, which can sometimes be more flexible than fixed sinusoidal patterns.

### Phase V: Causal Masking
1. **Why does GPT require masking?**
   To prevent information leakage from future tokens. During autoregressive generation, the model must predict token $t$ using only tokens $0$ to $t-1$.
2. **Why does BERT not use masking in this way?**
   BERT is designed to be a bidirectional encoder that uses context from both the left and the right to understand a word in context, so it doesn't need causal masking.

## Phase XIII: Error Analysis and Comparisons

### Part A: Bigram vs GPT
**Real-life Experimental Findings:**
We generated text from both our MiniGPT (trained on 3M chars) and a Bigram model (trained on the same 3M chars).
1. **Prompt:** `"Born in 1879, Albert Einstein was a"`
   - **GPT:** "Born in 1879, Albert Einstein was a professional football manager . He then moved to New England with an early 1909 – 07 season after he joined his career at the end of the League Cup..."
   - **Bigram:** "Born in 1879, Albert Einstein was a short frame . James <unk> to ward of the east of ordinary persons 15 @,@ 700 m"
   
2. **Prompt:** `"During the Second World War, the Allied forces"`
   - **GPT:** "During the Second World War, the Allied forces were trained with the and German troops . = = = General of Romanian Land Forces was born in January 1919..."
   - **Bigram:** "During the Second World War, the Allied forces . Mathews was supportive . Beginning in chess . Yue pick up great and blue became separated"

3. **Prompt:** `"The Mona Lisa is a painting by the Italian artist"`
   - **GPT:** "The Mona Lisa is a painting by the Italian artist , and it has been described as " admirrase " in the same name " . According to Itado 's review for her album..."
   - **Bigram:** "The Mona Lisa is a painting by the Italian artist Margaret <unk> , aided by Allied victory to the Aten . Everything else ."

**Explain:**
- **Context length:** The Bigram model completely loses the topic after 1 token. For example, in prompt 1, after "a", it predicts "short frame" because it only looks at the word "a", completely forgetting Albert Einstein. GPT remembers the biographical context over multiple sentences (e.g., "He then moved to... joined his career").
- **Dependency modelling:** Bigram breaks basic grammar rules constantly. In prompt 2, it ends the sentence abruptly after "forces", creating a fragment: "the Allied forces . Mathews was supportive ." GPT correctly identifies that "forces" is the subject and follows it with a verb phrase ("were trained with..."), maintaining grammatical dependency even if factually wrong.
- **Vocabulary generalisation:** Bigram hits sparsity issues frequently and hallucinates out-of-vocabulary tokens (like `<unk>`) because of raw frequency counts, failing to generalize to new combinations (like "Italian artist Margaret <unk>"). GPT uses subwords (BPE), so it does not hallucinates an `<unk>` token, and can creatively generate pseudo-words like "admirrase" based on its learned subword embeddings.
### Part B: HMM vs GPT

| Property | HMM | GPT |
| :--- | :--- | :--- |
| **Context Size** | Very small (typically strictly 1 previous state) | Large (context window via attention) |
| **Markov Assumption** | Strict (future is independent of past given present) | Relaxed (can attend to any past token) |
| **Training Objective** | Maximize joint probability (often EM or MLE) | Next-token prediction (Cross Entropy) |
| **Long-Range Dependencies**| Extremely poor | Excellent (via Self-Attention) |
| **Scalability** | Poor (matrices grow too large for complex states) | Extremely high (scales with parameters and data) |

### Part C: BERT vs GPT

| Property | BERT | GPT |
| :--- | :--- | :--- |
| **Architecture** | Transformer Encoder | Transformer Decoder |
| **Attention Type** | Bidirectional Self-Attention | Causal (Masked) Self-Attention |
| **Training Objective** | Masked Language Modeling (MLM) | Autoregressive Language Modeling |
| **Generation Capability**| Weak/Not designed for it | Excellent |
| **Typical Applications** | Classification, QA, NER | Text generation, Chatbots, Translation |

### Part D: Viterbi vs Autoregressive Decoding

1. **How Viterbi searches for globally optimal tag sequences:**
   Viterbi uses dynamic programming to keep track of the most probable path to each state at each time step. By looking backward from the end, it guarantees finding the single highest-probability sequence of tags.
2. **How autoregressive decoding generates sequences token by token:**
   It samples or greedily picks the next token based only on the current context, adds it to the context, and repeats. It cannot "go back" and revise previous tokens if it reaches a dead end.
3. **Why Beam Search can be viewed as a loose analogue of Viterbi search:**
   Beam search maintains $K$ active hypotheses (paths) at each time step, similar to how Viterbi maintains the best path to each state. While Beam Search isn't guaranteed to find the global optimum (unlike Viterbi), it approximates a global search far better than greedy decoding.


### Part E: Reflection
**Why does next-token prediction lead to language understanding?**
By forcing the model to continuously compress its context to accurately guess the next word, it is implicitly forced to learn the rules of syntax, semantics, factual knowledge, and logical reasoning embedded within the training data. Simply memorizing patterns isn't enough to achieve low loss on massive diverse datasets; the model must build an internal world model and understand concepts to correctly predict what word naturally follows a complex thought.
