---
title: "How a Pile of Random Numbers Learns to Talk"
description: "From training to post-training to inference"
date: 2026-07-11
tags: ["ai", "unpacking-ai", "training", "inference"]
canonicalURL: "https://www.pathtostaff.com/p/how-a-pile-of-random-numbers-learns"
cover:
  image: "/images/random-numbers-learns-to-talk/cover.png"
  alt: "How a Pile of Random Numbers Learns to Talk"
  hiddenInSingle: true
---

> *This was initially published on [Path to Staff](https://www.pathtostaff.com/p/how-a-pile-of-random-numbers-learns), but I bring it here as I write more about learning AI.*

Welcome back to Path to Staff! This is the final part of the Unpacking AI series, where we cover Training, Post-Training and Inference, and watch it all come together.

I also crosspost this to [my personal Substack](https://sidwyn.com/) where you can subscribe if you are interested in more technical posts like these. After this piece, we will return to regular programming on career growth on Path to Staff.

Here’s the series thus far:

1. **[The Hardware Behind AI](https://www.pathtostaff.com/p/unpacking-ai-the-hardware-behind).** Where we talked about hardware, including transistors, fabricators, and most famously the memory-compute bottleneck (why RAM is so expensive!).
2. **[Model Architecture](https://www.pathtostaff.com/p/everything-a-senior-engineer-needs).** Where we covered the “Attention is All You Need” paper, talked about tokens, embeddings, attention, and covered the transformer block. This is a must-pre-read if you haven’t read it. You’ll need to understand this in order for this piece to make sense.
3. **Training, Post-Training, and Inference.** *(This essay.)* In this piece, we dig into how the numbers inside a model get set, refined and finally return an answer to you.

I originally sketched this as five parts, but as I’ve gone through this, I’ve realized it makes more sense to combine the last three into one post. So here we go.

![cover](/images/random-numbers-learns-to-talk/cover.png)

## A model is just a pile of numbers

When we look at a model, say ChatGPT 5.6, [industry estimates](https://lifearchitect.ai/gpt-5/) put frontier models like it in the trillions of parameters. So for this article, let’s work with a model that has a trillion parameters. These trillion random weights sit in memory and are all randomly initialized at the start.

Now when I say weights, I also include both weights and biases. These trillion weights and biases control connection strengths and trigger neuron thresholds respectively.

When ChatGPT 5.6 is first created with these randomized numbers, if you ask it anything, it is going to return pure gibberish to you. That’s because all of these numbers were randomly initialized.

Now, the life of these numbers has three phases:

1. **Pretraining:** where we set the weights and biases.
2. **Post-training:** where we refine the weights, adjust biases.
3. **Inference:** where we read the weights and biases to answer a user’s prompt.

We’re going to cover each of them in detail. Strap in!

![01](/images/random-numbers-learns-to-talk/01.png)

## Pretraining

#### What is pretraining?

Pretraining is how these weights go from random values to their first real values.

Using thousands of GPUs over months, the model reads the entire Internet corpus. As it reads this data, it blindfolds itself by hiding the next word. It then asks itself, “Hey, what’s the next word?” And based on whether it gets it right or wrong, it adjusts its weights.

This happens over and over again until its loss function plateaus, or the training budget runs out.

By the end of this pretraining, the model learns human language. A variety of concepts compress into a fixed number of weights, which include:

1. **Language:** grammar, vocabulary, sentence structure
2. **World knowledge:** what’s the capital of X?
3. **Relationships:** how do ideas and themes connect?

#### Why do we need pretraining?

Before LLMs were popular, AI models were supervised. This means that humans labeled examples by hand. A human would label a review as positive and hand it over to the machines to learn. As noble as this work was, you could never label enough examples to cover all of human language. It was impossible.

Instead, researchers took another spin on this. Pretraining is *self-supervised*. This means that the text it reads is also its own answer key.

When the model hides the next word for itself to guess, no humans are required. And best of all, now the entire internet is training data.

#### The pretraining loop

Here’s how pretraining works in detail.

1. **Initialize random weights.**
2. **Forward pass:** the model predicts the next token in the sequence by calculating [self-attention](https://www.pathtostaff.com/p/everything-a-senior-engineer-needs?open=false#%C2%A7is-attention-all-we-really-need).
3. **Grade the prediction:** using cross-entropy loss, produce a grade by taking the negative log of the probability the model got it correct.
4. **Backpropagate:** figure out which of the 1 trillion weights to nudge and in which direction.

#### Defining cross-entropy loss

What is cross-entropy loss? Let’s go back to the example “The cat sat on the _”.

As the model guesses the next word, it assigns a confidence probability to every word in its vocabulary. The loss looks only at the confidence it gave the *correct* word.

This is **cross-entropy loss** because it measures how different the model’s guessed probabilities are from the absolute certainty of the true answer.

If the model was 90% sure the next word was “mat”, the penalty is small, say maybe 0.1. And vice versa: if it gave the word “mat” a 1% probability, the penalty is large. This penalty is then handed over to backpropagation, which refines the weights for the next round.

![02](/images/random-numbers-learns-to-talk/02.png)

#### How backpropagation works

Now for the fun part. We have calculated the cross-entropy loss and have 1 trillion weights to tune. Which weights do we tune?

This is what backpropagation is about. It walks backward through the network, using the calculus chain rule to assign each weight a *gradient*, which measures how much that specific weight contributed to the error.

It then sweeps backwards layer by layer.

This sweep costs about twice the compute of the forward pass, because every layer has to do two matrix multiplications (matmuls).

![03](/images/random-numbers-learns-to-talk/03.png)

1. **Gradients for its own weights.** This tells the layer how it should change. It takes just one matmul, where the incoming gradient is multiplied by the activations that we saved earlier during the forward pass.
2. **Gradients to pass downstream.** This tells the layer how much blame to forward to the layer below it. We need a second matmul here, multiplying the incoming gradient by the transpose of the layer’s own weights.

As it moves backward through the hidden layers, backpropagation works out how a tweak ripples all the way through to the final error.

#### Updating weights with gradient descent

With backpropagation, every weight now knows its share of the blame.

To then update the weights themselves, we need to utilize an optimization algorithm called **gradient descent**. This is where we take a small step in the direction the gradients point.

How much of a step do we take? The **learning rate** tells us so. This is a hyperparameter (aka tunable by the ML engineer), and is one of the most important dials in all of training.

Modern training runs don’t keep learning rates fixed either. They follow a schedule, usually a cosine decay curve that starts small, peaks, then comes back down to around 10% of its maximum by the end of the run.

***Why this rollercoaster schedule, you might ask?*** Because the model’s needs change drastically over the run.

At the start, the freshly randomized weights produce chaotic gradients, which means big steps this early would destabilize the entire network. So we warm up slowly.

Once things stabilize, the learning rate peaks and the model takes massive strides across the loss landscape, rapidly picking up broad concepts. Then near the end, a high learning rate would just violently overshoot the target, so we shrink the steps down and let the weights settle into the lowest point of error.

![04](/images/random-numbers-learns-to-talk/04.png)

#### The Chinchilla Paper

Now that we know how to modify these weights with backpropagation and gradient descent, let’s go back to the data the model first ingests. *How much data exactly is needed to train a model?*

In 2020, OpenAI’s [Kaplan paper](https://arxiv.org/abs/2001.08361) proposed the first scaling laws, which said that as compute grows, you should make the model vastly larger while only slightly increasing the data it sees.

In other words, pile on as many parameters as you can. The industry started rushing towards enormous models trained on relatively tiny datasets. That’s why you see models like OpenAI’s original [GPT-3](https://en.wikipedia.org/wiki/GPT-3) (175 billion parameters) or Meta’s [OPT-175B](https://arxiv.org/abs/2205.01068) trained on just 300 billion tokens. Even Google’s massive PaLM at 540B parameters was trained on only 780 billion tokens. By Kaplan’s math, these were supposedly correct, since the data should not increase as much as the parameters do.

However, things changed two years later. In 2022, DeepMind suspected these gigantic models were severely undertrained, so they ran the experiment properly this time, building 400 different baseline models across a wide range of sizes and data pools. Most importantly, they built their own 70B model, [Chinchilla](https://arxiv.org/abs/2203.15556).

Chinchilla soundly beat much larger models like the 280B Gopher (DeepMind’s own model) and the 540B PaLM, with just a fraction of the parameters.

Their refined math gave the industry a new rule of thumb:

> **The Chinchilla-Optimal Rule** says that for compute-optimal training, you should feed the model roughly **20 tokens for every 1 parameter** (e.g., a 70B model needs about 1.4 trillion tokens).

![05](/images/random-numbers-learns-to-talk/05.png)

## Post-training

So we’ve spent months and (likely) a small fortune pretraining a model. Now what do we have? An elite next-word predictor.

Ask it “What’s the capital of France?” and it might reply with “What’s the capital of Germany? What’s the capital of Italy?”

Now that’s a perfectly reasonable continuation if all you’ve ever done is read the internet, but pretty useless if you actually wanted an answer. A base model that’s pretrained will happily autocomplete your prompt or ramble on endlessly.

**Post-training is where it learns to become an assistant that follows instructions, tells good answers from bad ones, and aligns with human preferences.**

A standard post-training pipeline has two phases, ***Supervised Fine-Tuning (SFT)*** and ***Reinforcement Learning from Human Feedback (RLHF)***. There’s also an emerging third act which we’ll get to shortly.

![06](/images/random-numbers-learns-to-talk/06.png)

#### Phase 1: Supervised fine-tuning (SFT)

SFT is teaching by example. And it is an important first step before we can move on to other RL methods (like RLHF below).

We take the raw base model and fine-tune it on a bunch of input/output pairs. **These pairs teach it a specific behavior, tone or task.**

For instance, you can teach it to be a customer service assistant. Or to be a calculator.

The loss calculation mechanics in SFT should look familiar, because it’s exactly the same next-token prediction loss as pretraining: *using cross-entropy loss.* However this time, the loss is only calculated on the response, and not the prompt. This is because we don’t want it to memorize the prompts (or the questions themselves).

![07](/images/random-numbers-learns-to-talk/07.png)

#### Phase 2: Reinforcement learning from human feedback (RLHF)

Now, SFT teaches the model formatting and conversational behavior. However, the model still struggles with fuzzier ideals like truthfulness or being helpful.

That’s where RLHF comes in.

Instead of being fed one perfect answer, the model generates several candidate responses to a prompt. Human annotators then rank these outputs from best to worst.

As a ChatGPT user, you’ve probably done this labeling yourself without even realizing it.

![08](/images/random-numbers-learns-to-talk/08.png)
*Screenshot of an older version of ChatGPT, taken from [Reddit](https://www.reddit.com/r/ChatGPT/comments/1rsitj0/stop_chatgpt_from_giving_me_two_responses/).*

Using reinforcement learning, the model receives a mathematical reward for high-ranking answers and a penalty for the bad ones. Of course, this is also one of the most expensive steps since it requires humans to tune the models. But now, after we’ve covered all the major themes, all 1 trillion weights are actively optimizing for human preference, tone and safety.

![09](/images/random-numbers-learns-to-talk/09.png)
*The RLHF loop where the model writes, humans rank, and then weights get updated.*

Within RLHF, there are a few techniques. In a nutshell these are:

1. **Rejection sampling:** generate several responses, then keep only the winning samples and fine-tune on those.

2. **Proximal policy optimization:** introduce a penalty (known as the Kullback-Leibler penalty) which is multiplied by a coefficient and subtracted from the reward when tuning the new model. This keeps the new model from drifting too far away from the original.

![10](/images/random-numbers-learns-to-talk/10.png)

3. **Direct preference optimization:** directly optimize the language model using the chosen and rejected responses, without training any reward model at all.

![11](/images/random-numbers-learns-to-talk/11.png)
*Traditional RLHF trains a reward model and then runs PPO on top of it, while DPO goes straight from the preference data to the aligned model.*

#### The next frontier: RL on verifiable rewards

Now, the classic version of RLHF has a massive bottleneck.

Human preference is subjective, *expensive* to scale, and often rewards text that merely *sounds* plausible over text that is actually correct.

So the industry is shifting toward **verifiable rewards**, which are problems where the ground truth can be checked by a computer program instead of a human eye.

> For example, instead of a human reading an essay, a sandbox environment tests whether the model’s generated code compiles and passes all unit tests, or whether its step-by-step math lands on the correct final equation.
> Once you replace human judges with code executors and math verifiers, models can self-correct, “think” longer via reasoning loops, and scale their capabilities far beyond the limits of human annotation.

#### Measuring model helpfulness and safety using evals

As models progress through pretraining and post-training, we need objective benchmarks to measure how capable they actually are. You have likely seen the scorecard tables in release notes from frontier AI companies.

These are standardized lists of verified benchmarks, or “evals”, and they roughly fall into these buckets:

1. **Knowledge & Reasoning:** everything from broad general knowledge across dozens of subjects to graduate-level logic puzzles designed by PhDs ([MMLU](https://arxiv.org/abs/2009.03300) / [GPQA](https://arxiv.org/abs/2311.12022)).
2. **Math & Logic:** ranges from multi-step grade school word problems to competition-level algebra and calculus ([GSM8K](https://arxiv.org/abs/2110.14168) / [MATH](https://arxiv.org/abs/2103.03874)).
3. **Coding & Engineering:** measures whether the model can write code that passes unit tests, and navigate complex multi-file repositories to fix real bugs ([HumanEval](https://arxiv.org/abs/2107.03374) / [SWE-bench](https://www.swebench.com/)).
4. **Safety & Alignment:** how reliably the model catches and refuses dangerous requests, such as generating hate speech, planning cyberattacks, or building weapons (Red-Teaming / Refusal Evals).

Companies are creating new evals constantly, and the big labs even have entire eval divisions now. When an eval saturates (models start hitting 95%+), it gets hardened or updated.

![12](/images/random-numbers-learns-to-talk/12.png)

To keep things fair, it’s also important that benchmarks don’t leak into training sets, which is why the trajectory is moving towards private, frequently updated tasks.

## Inference

Alright, we’re done with pretraining and post-training. It’s time to move on to the last part of this article. If you’ve made it this far, congrats!

We’ve finished training the weights, so now how do we turn them into answering machines?

**Enter inference.** Inference happens when you type a prompt and the model “infers” it to return an answer. There are two major phases:

1. **Prefill:** processes the entire prompt in parallel, performing a matmul across all the tokens and storing the Keys and Values in the KV cache. We covered this in the [previous essay](https://www.pathtostaff.com/p/everything-a-senior-engineer-needs). This phase is **compute bound** (buy the best GPUs!)
2. **Decode:** generates the response one token at a time, and every single token requires reloading the model weights from HBM. This phase is **memory bound**. (Remember the [memory wall](https://www.pathtostaff.com/i/200814425/the-biggest-issue-memory-bandwidth) from Chapter 1?)

Now since there are two phases with two different bottlenecks, this means there are two obvious metrics to track.

Prefill is measured by TTFT (time to first token), while decode is measured by TPOT (time per output token). Sites like [Artificial Analysis](https://artificialanalysis.ai/) track these numbers across providers.

![13](/images/random-numbers-learns-to-talk/13.png)

## Optimizing Inference

Let’s see how to optimize inference. We’ll talk about two key technologies, batching and PagedAttention. As software engineers, we are already used to these concepts in system design!

#### Batching

Hauling 140GB of weights across the memory bus just to emit one token for a single prompt is extremely inefficient. What’s in play right now is continuous batching, where a scheduler processes all the active sequences together. As soon as a sequence emits an END token, a new queued request fills that slot. Check out BentoML’s [animation](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching) if you want to see how this works.

> In practice, OpenAI and Anthropic both have Batch APIs ([OpenAI’s](https://platform.openai.com/docs/guides/batch), [Anthropic’s](https://docs.claude.com/en/docs/build-with-claude/batch-processing)) that do offline batch inference, which combine latency-insensitive jobs. This reduces cost for both themselves and their customers.

#### PagedAttention

As the industry converged on continuous batching, a new problem popped up.

Each request’s KV cache grows like crazy, and at admission time we have no idea whether a response will cost us 10 tokens or 10,000. As such, a lot of KV memory ended up fragmented and over-reserved.

![14](/images/random-numbers-learns-to-talk/14.png)
*Remember defragmentation in your PCs? Well, the same solution can also be applied to LLMs.*

Operating systems solved this exact problem decades ago when they invented virtual memory. They chopped application data into fixed-size “pages” and mapped them to scattered pieces of physical memory (called “frames”) using a centralized page table.

![15](/images/random-numbers-learns-to-talk/15.png)

The vLLM team looked at the KV cache and realized it behaved exactly like a running program’s memory footprint. It’s dynamic, unpredictable, and rapidly growing.

So instead of treating the KV cache as a giant rigid block that has to sit together in VRAM, they built a virtual memory manager for the GPU, where:

- **Tokens became bytes:** fixed chunks of tokens (usually 16 per block) get grouped into pages.
- **The KV cache became RAM:** instead of allocating memory upfront based on a guess of the maximum response length, the system hands out a new 16-token physical block only when the model has actually generated enough text to fill the current one.
- **The lookup table:** just like an OS page table, a logical manager maps a user’s sequence to these scattered physical blocks across the GPU’s memory.

#### Further inference optimization

There’s a lot more to cover in inference optimization, and the [BentoML inference handbook](https://bentoml.com/llm/inference-optimization/) I linked actually does a really good job here.

Three optimizations you should know are prefix caching, speculative decoding (having a small draft model race ahead), and prefill-decode disaggregation.

1. **Prefix caching.** Requests share prompt prefixes constantly, whether it’s the system prompt, few-shot examples, or earlier turns of a chat. So we cache their KV pages once and reuse them, skipping that slice of prefill entirely. This is also why your API bill has cache-read line items at a fraction of the input price.

![16](/images/random-numbers-learns-to-talk/16.png)

2. **Speculative decoding.** A small draft model races ahead and proposes several tokens, then the big model verifies the whole run in one parallel pass and keeps the longest correct streak. Since verification preserves the exact output distribution, you get more or less identical quality at lower latency. An example is using Gemma-2-2B as a proposer and Gemma-2-9B as a verifier.

![17](/images/random-numbers-learns-to-talk/17.png)

3. **Prefill-decode disaggregation.** Remember how prefill is compute-hungry while decode is bandwidth-hungry? If you mix them on one GPU, a whale of a prompt will stall everyone else’s token stream during its prefill. The fix is simple here: run the two phases on separate pools that are sized independently, while handing off the KV cache between them. Hardware is now being designed around this split, like NVIDIA’s Rubin CPX, a prefill-specialized chip.

![18](/images/random-numbers-learns-to-talk/18.png)

Inference chip startups are also evolving like crazy in this space. Look at [Groq](https://groq.com/), [Etched](https://www.etched.com/) or [MatX](https://matx.com/). These companies are raising a lot of money to take on the giants (NVIDIA and Google) with their own inference chips. Definitely an interesting space to watch.

## TLDR

Let’s wrap this up. Hopefully this article made sense. If not, feel free to leave your comments below.

Here are 8 things to remember from today:

1. **A model starts as a pile of random numbers.** Weights and biases, randomly initialized, that return pure gibberish until they’re trained.
2. **Pretraining sets the weights.** The model guesses the next word across the entire internet, grades itself with cross-entropy loss, and repeats until the loss plateaus or the budget runs out.
3. **Backpropagation is an error assignment scheme.** One backward sweep tells all 1 trillion weights how much they contributed to the error, at about twice the compute of the forward pass.
4. **Chinchilla says 20 tokens per parameter.** Before this rule of thumb, the industry was training gigantic models that were severely undertrained.
5. **Post-training turns a predictor into an assistant.** SFT teaches by example, RLHF optimizes for human preference, and RL on verifiable rewards is the next frontier.
6. **Evals keep score.** Standardized benchmarks for knowledge, math, coding and safety, with the trajectory moving towards private, frequently updated tasks.
7. **Inference has two phases with two bottlenecks.** Prefill is compute bound and measured by TTFT, while decode is memory bound and measured by TPOT.
8. **Inference optimization is a systems game.** Continuous batching, PagedAttention, prefix caching and speculative decoding all squeeze more tokens out of the same GPUs.

Thanks for following along on this Unpacking AI series!

If you like this sort of technical content, subscribe to [my other Substack here](https://sidwyn.substack.com/). For Path to Staff, we’ll be returning to regular programming on career growth. Stay tuned.
