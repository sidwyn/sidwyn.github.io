---
title: "Everything a Senior Engineer Needs to Know About What's Inside an LLM"
description: "Learn what AI models are made of"
date: 2026-06-20
tags: ["ai", "unpacking-ai", "transformers"]
canonicalURL: "https://www.pathtostaff.com/p/everything-a-senior-engineer-needs"
cover:
  image: "/images/inside-an-llm/part2-cover.png"
  alt: "Unpacking AI Part Two: Data and Model Architecture"
  hiddenInSingle: true
---

> *This was initially published on [Path to Staff](https://www.pathtostaff.com/p/everything-a-senior-engineer-needs), but I'm bringing it here as I start to write more about learning AI.*

Welcome back to Path to Staff! This series is a little different from our usual programming. In this series, we're covering LLMs and AI in-depth. 

As an engineer, I never really had the time to understand AI's internals. But I've spent the past few weeks doing deep research to unpack it all.

As a reminder, this is Part Two of a five-part series:
1. **The Hardware Behind AI – And How It's Programmed.** Transistors, semiconductors, and fabricators. Learn about the big players (TSMC, Nvidia, ASML). The memory-compute bottleneck. And all the acronyms you always wondered about (TPU, ASIC, FPGA, CUDA, etc.)
2. **Model Architecture.** *(We are here.)* Learn about what models are made of. We'll cover the paper that started it all ("Attention is All You Need"), plus talk about transformers and diffusion models.
3. **Training**. The meat of teaching a model. How does pretraining work? What goes into it (backpropagation, optimizers, loss functions)? What scaling laws should we weigh before we kick off an expensive training run (up to hundreds of millions of dollars)?
4. **Post-Training & Alignment.** How does one guide a model once it's been taught? How do we apply safety? How do we benchmark and know the model got better? How do we evaluate a model's performance?
5. **Inference, Serving and Agents.** This might be the most familiar topic, since it's closest to you as an AI user. How does a model output its token and serve the result to you (SSE)? How do systems stay fair and fast? What tools are available (MCP, RAG, tool use) and how do agents work?

![part2-cover](/images/inside-an-llm/part2-cover.png)

## Recurrent Neural Networks are No Longer Popular

I remember taking CS:188 by Pieter Abbeel at Berkeley, learning about Recurrent Neural Networks (RNNs) in 2014. I wish I'd paid more attention in class, but it was still a good foundation for working on this chapter. 

To my surprise, RNNs are less popular today. And there's a good reason for that.

If you haven't studied neural networks before, you should know that a neural network looks at an input, guesses what it is, then immediately forgets it. The simplest form is a *feed-forward neural net*, where the data goes through and outputs as a layer at the end.

Recurrent neural networks take this one step further, with a built-in memory loop. It looks at the first word, jots it down in a hidden notebook (or layer), and then reads the second and jots it down again. This repeats over and over again. The downside is that it has terrible long-term memory, and would only remember what was most recently fed. 

![2-rnn-decay](/images/inside-an-llm/2-rnn-decay.png)
*An RNN reads "The cat sat on the mat" one word at a time - its memory of "The" has almost faded by "mat".*

Upgraded versions of RNNs called [LSTM (Long Short-Term Memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRUs (Gated Recurrent Units)](https://en.wikipedia.org/wiki/Gated_recurrent_unit) were implemented to highlight important memories, but there were still two key problems with these recurrent neural networks. 

They were:
1. ***Sequential bottleneck:*** You can't compute the next step in a recurrent neural network till the current operation is complete. This means there's no way to parallelize training. A sequence of length N takes N sequential steps no matter how much hardware you throw at it. Training time was bound by the length of the chain, not the size of the chain. 
2. ***Long-range decay:*** information from early tokens fades before it reaches later ones, as seen in the image above. Training an RNN means backpropagating the signal through every step, multiplying by the same weights each time. So over a long sequence the gradient vanishes to zero or explodes, and the model can't learn that an early token should shape a much later one. 

## Enter Transformers

In 2014, [Bahdanau et al.](https://arxiv.org/abs/1409.0473) added a different concept to neural networks, called **attention**. Rather than force the whole input through one fixed summary, let the model look back at every input token and weight the ones that matter for the current step. Simplified, when generating a new token, the model looks back at all the previous tokens to generate a vector for each of them.

It worked well, and took an important step toward the transformer. However, it was still bolted onto a recurrent network, so the sequential bottleneck remained.

Three years later, [Vaswani et al.](https://arxiv.org/abs/1706.03762) made a large leap: keep the attention, **drop the recurrence entirely**. Their insight was that self-attention, where every token computes its relationship to every other token through dot products, solves both problems (sequential bottleneck & long-range decay) at once. 

With matrix multiplication happening over all pairs simultaneously, the whole sequence can now be processed in parallel rather than step by step. *And* with no recurrent chain, there's no information left to decay. 

![3-sequential-vs-parallel](/images/inside-an-llm/3-sequential-vs-parallel.png)
*The same five tokens, two ways to mix them. The RNN passes information hand-to-hand down a chain, so it runs one step at a time and the earliest signal fades by the end. Self-attention connects every token to every other at once — fully parallel, with nothing left to decay.*

Eight Google researchers laid this out in a paper titled "**Attention Is All You Need**". The architecture they proposed within is called the ***transformer***.

At first, they built it to improve machine translation, but they already saw further, noting in the paper that the same architecture should extend to other tasks. They were not wrong.

## An Overview of Transformers
A transformer is a neural network that processes sequences. 

An input goes in, passes through N identical blocks and a prediction head converts it into an output. 

> A prediction head is a translator at the end of the model that turns the model's numbers into a score for every word. These raw scores are called *logits*.

There are different types of transformers. Let's first look at an example of translating a sentence. You have an input that gets read, which gets translated to an output.

Say you want to translate "The cat sat on the mat" to Chinese. This goes from "The cat sat on the mat" to "猫坐在垫子上". 

In this case, we are using an encoder-decoder architecture. The encoder reads the input, while the decoder generates output.

Here's what the architecture looks like:
![4-encoder-decoder](/images/inside-an-llm/4-encoder-decoder.png)

Now remember that the authors were *solving machine translation*. However, the AI field took this architecture apart and found that the halves were independently useful! Google used the encoder, trained it to understand text, and got [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers), released in 2018. 

By 2019, it was rolled into the ranking backend of Google Search. It was able to understand each word in context with all other words simultaneously. It was incredibly good at understanding context, and so most researchers would use it for sentiment analysis, or for named entity recognition.

Google also worked on decoders, but it worked towards [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198). Then Alec Radford at OpenAI took the same move, but in a different, important direction: combining the decoder-only transformer with his generative-pretraining thesis. 

In [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Alec et al. argue that raw text is everywhere, but text that's been labeled for a specific task (like "this review is positive") is rare and expensive, which limits models that can only learn from labeled examples. 

To fix this, let a model learn language broadly by reading a huge pile of unlabeled text, then give it a small amount of labeled data to specialize on a particular task. That two-step recipe produces big improvements.

#### Enter GPT

These two steps are known as:
1. ***Generative pre-training*** (GPT, anyone?) - read unlabeled text
2. ***Fine-tuning*** - specialize on a particular text

This generative pre-training is the first step of an AI model, where it is trained on a massive general dataset to learn general-purpose representations and patterns (think grammar, language, reasoning). 

Alec, together with Karthik Narasimhan, Tim Salimans and Ilya Sutskever, came up with **GPT-1** in June 2018, a decoder-only transformer. 

While Google was doubling down on BERT, OpenAI doubled down on Generative Pre-Training (GPT), pushing out [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in 2019 with the same architecture but roughly 10x the parameters and data. Now, the model could start doing tasks zero-shot from prompting, without any fine-tuning. [GPT-3](https://arxiv.org/abs/2005.14165) in 2020 scaled another 100x.

> What happened to the fine-tuning step? By GPT-3, the model showed that it mostly didn't need it. A big enough model was smart enough to understand most tasks from a prompt alone. More work was put into post-training, which we will cover in a future chapter.

#### Encoder Only vs Encoder-Decoder Models

Now, it is important to understand why GPTs (decoder-only transformers) became much more popular than say the encoder-decoder or the encoder model. 

Three reasons:
1. **A decoder trains on predicting the next token.** This requires no labels and task-specific setup. This meant that the entire internet corpus becomes training data.
2. **The objective was universal.** You could use it for translation (e.g. "continue this text"), summarization ("summarize this text"), etc. All of these reduced to continuation. On the other hand, BERT could never be prompted this way.
3. **Why it beat encoder-decoder models:** There was just no input sometimes. Inputs work for translation, but there's no clear input in a multi-turn conversation. Having both an encoder and a decoder also meant cross-attention added more surface area to the model. This also resulted in more ways for a training run to go wrong.

> If you want to see how far the decoder family tree has branched — LLaMA, Mistral, DeepSeek, Qwen, and the architectural deltas between them — Sebastian Raschka maintains an excellent [visual gallery of LLM architectures](https://sebastianraschka.com/llm-architecture-gallery). 

From here on, we'll zoom into the decoder-only stack - the architecture behind GPT-style models - and walk through what happens inside it, one component at a time.

## Turning Text into Numbers

> In this section, we'll also learn why the model keeps getting "how many r's in strawberry" wrong. Yes, it has to do with tokenization!

If a transformer operates on sequences of vectors, how do we turn text into numbers? How does "The cat sat" turn into vectors? This is also known as tokenization, because we are breaking down a sentence into individual tokens. 

What are simple ways to break up a sentence into tokens? Here's a few:
1. ***Words***: "The", "cat", "sat" are all valid words in the dictionary. However, unknown words have to be represented by "< UNK >", where UNK stands for unknown. And when we decode and encode that word, we lose precision.
2. ***Characters***. Every character gets represented, however it is a lot longer than what we need.
3. ***Subwords***. Can we break up a word like "misrepresented" as "mis", "represent" and "ed", to individual tokens? This is the key idea behind Byte-Pair Encoding, which was actually invented in 1994.

BPE was invented by Philip Gage for data compression, and was [brought to NLP in 2016 by Sennrich et al.](https://arxiv.org/abs/1508.07909) for machine translation. 

Let's take a look:
![5-byte-pair-encoding](/images/inside-an-llm/5-byte-pair-encoding.png)
*BPE on our sentence: after three merges, "at", "th" and "the" have joined the vocabulary. Real tokenizers keep going to ~32k-200k.*

The steps are simple:
1. Start with a base vocabulary of individual characters.
2. Count every adjacent pair of tokens
3. Find the most frequent pair, and merge it into a new token, adding it to the vocabulary.
4. Repeat step 2-3 until the vocabulary hits your target size (which is a model hyperparameter, typically around 32-200k).

Now, BPE has evolved a little more than that into byte-level BPE. In GPT-2, BPE is run over raw bytes, so that the base vocab is exactly 256 symbols and every possible string (across languages) is just some sequence of bytes.

#### Where BPE falters

There are a couple of places where BPE falters. Here's three of them:

1. ***Token spelling:*** "How many r's are there in strawberry?" This is hard for a model because it could have broken out "straw" and "berry" into two different tokens, say token 2516 and 426. There would be no way for the model to know the spelling of token 426.

*How this was fixed:* Reasoning (model actually decomposes the word), memorization (remembering that strawberry has 3 'r's), and tool use (model writes code to count the number - do you see ChatGPT or Claude sometimes suddenly writing code to answer a math problem? Well, this is why.)

2. ***Arithmetic:*** Asking a model to do math is hard because digits are arbitrarily chunked. This was fixed by splitting digits into their own tokens. 

3. ***Multilingual cost inequality:*** While in English tokens cost roughly 4 characters each, other languages may cost more tokens per character. This is definitely trickier.

*How this was fixed:* Well, it's still ongoing, but OpenAI's GPT-4o tokenizer (o200k) doubled the vocabulary to 200k tokens and dramatically reduced token counts for non-English languages, letting them fit more words into the mix. 

There's also been new advances in this space, such as [ByT5](https://arxiv.org/abs/2105.13626) and the [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (Meta, 2024), which operate on raw bytes, learning to group bytes into variable-size patches. This is an interesting frontier, but for the sake of this article, will be left as extra reading.
## From Token IDs to Meaning: Embeddings

**So tokenization gives us integers.** But a token ID like 1024 is just a row number — it doesn't *mean* anything by itself. 

The thing that gives it meaning is a giant lookup table called the ***embedding matrix***: one row per vocabulary entry, where each row is a long vector of numbers. 

The length of that row is the model's hidden size - in many 7B-class models, that's **4,096 numbers** per token (our running example uses the original paper's 512). That's a crazy large number, but if you think about it, it could actually encode a lot of information.

> **Can a model encode too much information?** Yes, in the sense of waste. A wider hidden size gives each token more room to store features, but past a point you get diminishing returns, where extra dimensions end up redundant or near-empty.

Let's work through an example.

When the tokenizer hands the model the ID for "cat", the model looks up that row and works with the vector instead. Here's the magical part... nobody designs these vectors! 

They're learned during training, and semantically similar tokens end up close together in space - "cat" lands near "kitten", "mat" near "rug". The geometry even supports arithmetic: the [famous example](https://arxiv.org/abs/1301.3781) is king − man + woman ≈ queen.

![6-embeddings](/images/inside-an-llm/6-embeddings.png)
*In this image, we've squashed our vector size to 2-dimensions. Take note that there are up to 4,096 dimensions as mentioned above.*

However, there's one thing the embedding does *not* carry: position within a sentence. The vector for "cat" is identical whether "cat" is the first word of your prompt or the fiftieth. 

## Positional Encoding

Now in order for encoders and decoders to understand order in a sentence, we have positional encoding. This is placed at the start of the Encoder and Decoder stacks, immediately after raw tokens are converted into word embeddings, and before the Attention layers.

![7-positional-encoding](/images/inside-an-llm/7-positional-encoding.png)
*Where positional encoding sits in the stack - and what gets added to "cat".*

Now for our same example, we need to distinguish between "The **cat** sat on the **mat**". and "The **mat** sat on the **cat**". To do this, we need each word's position! For instance:

- Position 0: **"The"**
- Position 1: **"cat"**
- Position 2: **"sat"**
- Position 3: **"on"**
- Position 4: **"the"**
- Position 5: **"mat"**

The model handles this by generating a unique *positional vector* for each index and merging it with the *word vector*.
1. ***Positional Vector:*** A 512-length vector is generated specifically for *Position 1*. We used sine and cosine waves in 2017 since they would fluctuate strictly between -1 and 1 and would not overpower the actual meaning of words.
2. ***Word Vector:*** The word **"cat"** is converted into a 512-length vector representing its semantic meaning.
#### Limitations of Sine & Cosine Waves
However, there were limits to these sine and cosine waves for positional vectors. There were three key reasons:
1. ***Corruption of word semantics:*** The sinusoidal method adds position directly into a token's meaning. We'd rather have a way to encode position into a vector's angle, preserving its length and not distorting the original features.
2. ***Generalized sequence lengths:*** Positions are limited by the inputs. E.g. if you trained an LLM on 4000 token limits, it would not understand positioning for slots 4001 and beyond. 
3. ***Relative text shifts:*** Sometimes the same grammatical phrase occurs at different places in a text. The math with positional encoding leaves behind absolute room numbers that change the final score based on where the phrase sits in the document. Often all we want is the relative position, not the absolute one.

We look at [Rotary Positional Encoding](https://arxiv.org/abs/2104.09864) (RoPE, proposed by Su et al. in 2021) to solve this. It gets a little deep here, so I'll try to simplify: Instead of changing a word's vector by _adding_ position numbers to it, RoPE injects order by physically **rotating** the vectors in a mathematical space based on their position index.

![8-rotary-positional-encoding](/images/inside-an-llm/8-rotary-positional-encoding.png)
*RoPE turns position into rotation: the hand's length (meaning) never changes, only its angle (position).*

Think of each word as a clock hand. The hand's length is the word's meaning which never changes. Its angle is the position: "cat" at position 1 gets rotated a little, "mat" at position 5 gets rotated more. 

Now this way, we have a much better way to represent token positions.

## Is Attention All We Really Need?

In 2017, eight researchers at Google were racing to finish a paper for NIPS, the field's biggest conference, before the deadline. The work was done. The architecture that would reorganize all of AI was sitting in the draft. What they didn't have was a title.

Then, Llion Jones, the Welsh researcher among them, threw one out almost as a joke: after the Beatles' "All You Need Is Love," why not "Attention Is All You Need"? He didn't expect the team to keep it, but — they kept it! 

And the joke turned out to be the most honest summary. Where everyone else was bolting attention onto recurrent networks, this paper deleted the recurrence and kept only attention. 

> **Heads up:** There's going to be a lot of math in the next few sections. Take your time to digest them, and use your favorite AI to break it down for you. I recommend understanding this formula in-depth because it *is* the basis of today's LLM models.

Let's now take a look at the key paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Most importantly, look at this formula in Section 3.2.1: Scaled Dot-Product Attention:

![Screenshot 2026-06-12 at 9.42.31 AM](/images/inside-an-llm/Screenshot%202026-06-12%20at%209.42.31%20AM.png)
I know this formula looks scary if you haven't seen it before, but let's walk through it together with an example.

The key thing to note is that instead of RNNs reading one word at a time, we are going to have a sentence where each token looks at every other token.

Back to our favorite example: The cat sat on the mat.
![10-self-attention](/images/inside-an-llm/10-self-attention.png)
*"sat" attends to every token. If you look at those with the highest scores, it's what's relevant to "sat". For instance: who sat? "cat" (0.42). Sat where? "mat" (0.30).*

Now each token gets broken down into three vectors:
1. ***Query:*** What am I looking for from other tokens?
2. ***Key:*** What can I give to other tokens looking at me?
3. ***Value:*** What gets passed along when I match with other tokens?

Remember matmuls from the first chapter? Here's where they come into play. Every token's Query is going to be dot-producted with the Key of each other token, to produce a score. These are all values that have been initialized to random weights.

This calculation is QK<sup>T</sup>, where K<sup>T</sup> just means the K matrix transposed (flipped along its diagonal). We do this to produce a similarity score.

![11-qkv-scores](/images/inside-an-llm/11-qkv-scores.png)
*Step 1: "sat"'s Query dot-producted with every Key, then divided by √d<sub>k</sub>.*

We then divide by √d<sub>k</sub>. This refers to the key dimension (e.g. if the Key is [0.1, -0.5, 0.9, 0.2], the dimension is 4, and the sqrt of that is 2). 

> The reason we do this is that dot products of bigger vectors produce bigger numbers: the variance of the dot product of two random d<sub>k</sub>-dimensional vectors grows to d<sub>k</sub>, so dividing by sqrt(d<sub>k</sub>) brings the scores back to a sane scale before they hit the softmax.

Then we need to convert these scores to probabilities. 

To do this, we use the softmax function. Softmax exponentiates each score and normalizes. 

> The reason for the exponent is to eliminate negative numbers, since e raised to any power is always positive.

![12-softmax](/images/inside-an-llm/12-softmax.png)
*Step 2: softmax exponentiates and normalizes - each row becomes a probability distribution.*

Now last but not least, we have to multiply the weights by the Value matrix. This focuses the model on the most compatible and valuable keys, producing a context-aware representation of each token.

![13-attention-values](/images/inside-an-llm/13-attention-values.png)
*Step 3: each token's Value is scaled by its attention weight and summed - "cat" and "mat" dominate the blend that becomes the new vector for "sat".*

The best part of all is that this is all happening in parallel, which is why GPUs have become so popular.
#### Multi-Head Attention

Now to take this even more parallel, we can split up the vector dimension into multiple smaller vectors. Each of these smaller vectors is a head. In the paper, the model's total vector dimension was set to 512, split into 8 heads of 64 dimensions each. Each head learns to look for something different: one might track which noun a word refers to, another might track syntax.

![14-multi-head-attention](/images/inside-an-llm/14-multi-head-attention.png)
*The 512-dim vector is split into 8 heads of 64 dims each. The heads run in parallel, then their outputs are concatenated back into a single 512-dim vector.*

That 64 turns out to be a very hardware-friendly number. Remember GPU warps from the first chapter? A warp is a group of 32 threads executing in lockstep, so a 64-dimensional head divides cleanly across warps. These days, AI engineers choose the head dimension - usually 64, 128 or 256 - to align with the GPU's Tensor Cores.

> What do thousands of heads actually learn? Anthropic's interpretability team found [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - heads that spot the pattern "A B ... A" in your prompt and predict B comes next. They're one of the clearest known mechanisms behind in-context learning, where the model picks up a pattern from your prompt and continues it.

![15-induction-heads](/images/inside-an-llm/15-induction-heads.png)
*An induction head spots that the current token already appeared earlier, then predicts whatever followed it last time - "A B ... A" → B.*

#### Masking Matrices
Now in the last part of Section 3.2.3 in the paper, there's a key sentence here: "*We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.*"

We have another step here: to apply a mask matrix (kind of like a mask in Photoshop). 

This ensures that **only past and present words** can be seen, and that the model cannot cheat by looking ahead during training. This is also known as *autoregression* in statistics.
![Screenshot 2026-06-12 at 10.07.28 AM](/images/inside-an-llm/Screenshot%202026-06-12%20at%2010.07.28%20AM.png)

#### Why The Context Window Matters
Here's a crucial property of attention: its computational cost scales quadratically, O(N²), with the sequence length. E.g. if you double your prompt from 100 to 200 tokens, the compute and memory for attention quadruples.

This is why context windows exist at all. Every new token has to attend to every token before it, and the model has to keep all those Keys and Values around in memory (the "KV cache"). 

With RoPE, there's a second ceiling: positional coordinates only stretch as far as you've trained them. A model trained on 4,000-token sequences has simply never seen a rotation angle for position 50,000. So the context window is bounded by both compute and the positional encoding's training range.

> One more wrinkle: even *within* the window, attention isn't spread evenly. Researchers documented a "[lost in the middle](https://arxiv.org/abs/2307.03172)" effect - models use information at the start and end of long prompts far more reliably than information buried in the middle. This is why prompt tips like "put the important context first" or "repeat the key instruction at the end" work well.

## The Other Important Blocks
Let's go back to the transformer architecture. Attention gets all the headlines (and the paper title), but if you look at the diagram, there are still several other important blocks.

![17-block-anatomy](/images/inside-an-llm/17-block-anatomy.png)

#### Add & Norm
This pair shows up after every attention unit and every feed-forward unit, and it does two jobs.

Adding: the block's output (the residual) is added back to its input. This is to make sure that gradients don't vanish halfway through a deep stack (just like how they did in RNNs). Think of it as a gradient highway: even if one layer learns nothing useful, the signal can flow straight through the addition unharmed. This trick came from [ResNet](https://arxiv.org/abs/1512.03385) in computer vision, and it's a big part of why we can train models hundreds of layers deep.

Normalization: after all that adding, we rescale each token's vector to mean 0 and variance 1, to make sure we don't blow numbers out of proportion:
 
> The formula, if you were interested, is:
> $$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$
> Reading it left to right: $x$ is the token's incoming vector. $\mu$ is the mean of its values and $\sigma$ their standard deviation, both computed *per token* across the vector's own dimensions (not across the batch).  Subtracting $\mu$ then dividing by $\sigma$ re-centers the vector to mean 0 and variance 1. 
> 
> But forcing every vector to look statistically identical is too rigid, so we hand back two *learned* knobs: $\gamma$ (gamma) rescales the spread and $\beta$ (beta) shifts the center, and the model figures out the right values for each during training. 
> 
> Modern models like Llama use a leaner cousin called [RMSNorm](https://arxiv.org/abs/1910.07467) (skip subtracting the mean, just divide by the magnitude), and apply it *before* each unit rather than after - "pre-norm" - because it trains more stably.

#### Feed-Forward Network
In this unit, each token digests what happened during attention. It takes its current vector and blows it up to 4x the width, before squashing it back down:

![18-feed-forward](/images/inside-an-llm/18-feed-forward.png)
*The FFN blows each token's vector up 4x, bends it, and squashes it back. No token-to-token communication.*

Why do we do this? When we blow up the vector, we can analyze it across many more dimensions. Think of a detective who empties a cramped case folder out across a huge table: in the original 512 dimensions, clues are stacked on top of each other and hard to tell apart, but spread across 2048 dimensions each pattern gets its own bit of room. 

Now, there is no token-to-token communication here. This all happens within one token.

That GELU (Gaussian Error Linear Unit) in the middle is the activation function. Modern models like Llama, Mistral and PaLM have since moved on to [SwiGLU](https://arxiv.org/abs/2002.05202), a different variant.

Here's the part that surprised me: **this boring block is where most of the model's parameters live.** 

For the original 512-dimension transformer, $W_1$ is 512x2048 and $W_2$ is 2048x512 - about 2.1M parameters per layer, roughly double what attention uses. 

In modern LLMs, the feed-forward layers hold around two-thirds of all weights. There's [research suggesting](https://arxiv.org/abs/2012.14913) they act as key-value memories - this is plausibly where "Paris is the capital of France" is stored.


> Formula for those who are keen: 
> $$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)\,W_2 + b_2$$

#### The Final Linear Layer
After the final block, each token's vector gets multiplied by one last matrix (the "LM head"), producing one score *per vocabulary entry* - if your tokenizer has 200k tokens, that's 200k scores. 

These scores are called **logits**, and a final softmax turns them into next-token probabilities. We'll look at this in-depth in the next essay.

## The Attention Formula

Let's wrap up. How do we write Attention in code?

Well, good news. The entire attention formula fits in about a dozen lines of NumPy. Andrej Karpathy also covers this in a [longer video](https://www.youtube.com/watch?v=kCc8FmEb1nY), but here it is:

```python
import numpy as np

tokens = ["The", "cat", "sat", "on", "the", "mat"]
d_k = 4
np.random.seed(0)
E = np.random.randn(len(tokens), d_k)          # pretend embeddings
Wq, Wk, Wv = (np.random.randn(d_k, d_k) for _ in range(3))
Q, K, V = E @ Wq, E @ Wk, E @ Wv

scores = Q @ K.T / np.sqrt(d_k)                # QKᵀ / √dk
mask = np.triu(np.ones_like(scores), k=1)      # hide the future
scores = np.where(mask == 1, -np.inf, scores)
weights = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)  # softmax
output = weights @ V                           # weighted sum of Values

print(np.round(weights, 2))   # each row: where that token is looking
```

**That's the transformer!**

Next time you watch tokens stream out of Claude, you'll know what's happening underneath: bytes merging into tokens, vectors rotating by position, and every word attending to every word that came before it.

---

## Extra Reading Section!

Because the entire essay is so long, I've left the following in an appendix here. I know most of you won't get to these, but they are very useful for understanding modern architecture.

#### Grouped-Query Attention

Back to Attention for now. When the model generates the token 5,001, it needs the Keys and Values of all 5,000 tokens before it. Recomputing them every step would be madness, so they're cached - per layer, per head. At long contexts, this cache can eat more VRAM than the model weights themselves.

![19-grouped-query-attention](/images/inside-an-llm/19-grouped-query-attention.png)

The modern fix is [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA): instead of every head keeping its own Keys and Values, groups of query heads share one K/V head. Llama 2 70B runs 64 query heads against just 8 shared K/V heads; Mistral 7B does 32 against 8. You keep almost all the quality of full multi-head attention with a fraction of the cache - which is why nearly every open model since 2023 ships with it.

#### FlashAttention

We'll look into ways to optimize this through [FlashAttention](https://arxiv.org/abs/2205.14135), invented by Tri Dao et al. in 2022. Remember the memory bottleneck that we explored last chapter? GPUs have a lightning-fast but tiny SRAM, and a big but slow VRAM (HBM).

![20-flashattention](/images/inside-an-llm/20-flashattention.png)

The naive implementation computes the full N×N score table for "The cat sat on the mat", writes it out to slow VRAM, reads it back in to do the softmax, writes it out again... Now, the matrix isn't the bottleneck. The memory traffic is. How do we solve this?

With FlashAttention. The trick here is to use tiling. The sentence gets cut into small blocks, and each Streaming Multiprocessor (SM) grabs a tile - say, the Queries for "The cat" against the Keys for "the mat" - and computes that entire piece inside its local SRAM without ever writing the intermediate table out to VRAM. To keep the math correct, it accumulates running statistics locally and applies the softmax normalization exactly once at the end of each row (this is called the *online softmax*).

The result is the exact same answer as standard attention, just with far fewer round trips to slow memory. No approximation, just better traffic management.

There's also been further headway in this space, which I will leave as further reading:
- ***Linear Attention & State Space Models ([Mamba](https://arxiv.org/abs/2312.00752)):*** Architectures trying to replace the attention formula completely to achieve \(O(N)\) linear scaling.
- ***RoPE Scaling ([YaRN](https://arxiv.org/abs/2309.00071) & RoPE Interpolation):*** The mathematical tricks used to compress and stretch coordinate systems, letting a model trained on 4,000 tokens seamlessly read 128,000 tokens.

#### Mixture of Experts (MoE)
There's a newer technique that turns a dense transformer block into a sparse network. This increases the total parameter count by hundreds of billions of weights without increasing the FLOPs spent per token. How do we do this?

We swap out the single Feed-Forward Network for a fleet of them - the "experts" - plus a tiny router in front.

![21-mixture-of-experts](/images/inside-an-llm/21-mixture-of-experts.png)
*The MoE router scores all 8 experts and wakes only the top 2 - the rest of the parameters sleep.*

When a token vector enters the MoE layer, the router multiplies the token vector by its own weight matrix to score every expert, then routes the token to the top-k highest-scoring experts. Only those experts run; their outputs are summed, weighted by the router's scores. Not all the experts are activated at the same time - that's the entire trick.

The math here:

$$y = \sum_{i \,\in\, \text{TopK}} g_i(x)\, E_i(x), \qquad g(x) = \text{softmax}(x \cdot W_g)$$

So when "cat" passes through, maybe expert 3 (which has drifted towards noun-ish, animal-ish patterns during training) and expert 7 fire, while the other six sleep. You get a model with trillions of potential parameters where each token only pays for a fraction of them.

The idea dates back to [Shazeer et al. in 2017](https://arxiv.org/abs/1701.06538) (yes, the same Noam Shazeer from Attention Is All You Need - and yes this guy [just moved to OpenAI](https://x.com/NoamShazeer/status/2067400851438932297) this week.), scaled up by Google's [Switch Transformer](https://arxiv.org/abs/2101.03961). Today, [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) (8 experts, top-2: 47B total parameters, only 13B active per token) and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (256 experts plus 1 shared, top-8: 671B total, 37B active) are leveraging this.

One catch worth knowing: routers can collapse, learning to send everything to a couple of favorite experts while the rest atrophy. The fix is a load-balancing loss during training that nudges tokens to spread out evenly.

---