---
title: "Unpacking AI: The Hardware Behind AI"
description: "A new series on learning AI, from the ground up"
date: 2026-06-06
tags: ["ai", "unpacking-ai", "hardware"]
canonicalURL: "https://www.pathtostaff.com/p/unpacking-ai-the-hardware-behind"
---

Welcome back to Path to Staff. I recently left Meta for personal reasons (not laid off!), and have found much more time to write. This means learning as much as I can and distilling what I learn into these articles. 

Now back to the topic: As an engineer who never really interfaced much with AI, I realized I didn't know much about it at all. Since I’ve more time now, I've spent the past few weeks diving really deep to understand AI from the bottom up. And I want to share those learnings with you today. 

This deep dive series is a five-part series:
1. **The Hardware Behind AI – And How It's Programmed.** Transistors, semiconductors, and fabricators. Learn about the big players (TSMC, Nvidia, ASML). The memory-compute bottleneck. And all the acronyms you always wondered about (TPU, ASIC, FPGA, CUDA, etc.)
2. **Data & Model Architecture.** Learn about how models are made of. We'll cover the paper that started it all ("Attention is All You Need"), plus talk about transformers and diffusion models. And of course, we'll cover how training data is prepared for these models (what sources? how is the data decontaminated and filtered?)
3. **Training**. The meat of teaching a model. How does pretraining work? What goes into it (backpropagation, optimizers, loss functions)? What scaling laws before we kick off an expensive training run (up to hundreds of millions $)?
4. **Post-Training & Alignment.** How does one guide a model once it's been taught? How do we apply safety? How do we benchmark and know the model got better? How do we evaluate a model's performance?
5. **Inference, Serving and Agents.** This might be the most familiar topic, since it's closest to you as an AI user. How does a model output its token and serve the result to you (SSE)? How do systems stay fair and fast? What tools are available (MCP, RAG, tool use) and how do agents work?

Over the course of this series, I expect the syllabus to change as I learn more about AI. I also welcome questions in the comments, since this will help me tweak it. If there are several acronyms, I also list their definition at the top of the section. 

Quick note: As much as I love using AI in my work, most of these will be written by hand, with light editing by AI. If you're interested in reviewing drafts, please also reply to this post and let me know!

---
## Transistors And Their Importance

*At a glance*

| Acronyms | Definition                                                                                         |
| -------- | -------------------------------------------------------------------------------------------------- |
| EUV      | Extreme Ultraviolet - type of lithography (manufacturing) used to print intricate scale onto chips |
| ASML     | Advanced Semiconductor Materials Lithography - Dutch company that makes EUV machines               |
| TSMC     | Taiwan Semiconductor Manufacturing Company, leading semiconductor factory based in Taiwan          |
To understand Artificial Intelligence we first have to go to the core of it.
 
AI runs off GPU chips. These chips are made from transistors which are manufactured using EUV machines. 

![Screenshot 2026-05-30 at 8.07.37 AM](/images/unpacking-ai-hardware/Screenshot%202026-05-30%20at%208.07.37%20AM.png)

Let's break each of these down, starting with a transistor. 

A transistor is a semiconductor device that controls the flow of electricity. It uses a small electrical signal at one terminal to control a much larger current. It either (1) boosts a signal, or (2) decides whether a current can pass. In other words, it acts as either an amplifier or switch. *A semiconductor, most commonly silicon, is a material that conducts electricity only under certain conditions. Its conductivity can be modified by adding impurities.*

**Who designs these chips?** Nvidia and AMD are the biggest players when it comes to designing chips. Not far behind is Google, Amazon and now Meta when it comes to chip design. However, these companies operate as "fabless" designers. That means they *only* build the architecture, while outsourcing the physical production which costs up to $20B for a foundry. These foundries require tens of billions a year in capital spending to stay current, a cost that most of these companies are not interested in.

**Who then, makes these transistors?** TSMC (Taiwan Semiconductor Manufacturing Company) does. But they require special machines, called extreme ultraviolet (EUV) machines, and a process called lithography, which is the act of printing on chips. There are other foundries like Samsung and Intel, but they are not as advanced as TSMC, which currently holds 70% of the foundry market.

**Who makes these EUV machines?** These are only being manufactured by ASML, which has a monopoly foothold in the EUV machine industry. China is fast catching up, but is still roughly 5 years behind.

![Pasted image 20260527093253](/images/unpacking-ai-hardware/Pasted%20image%2020260527093253.png)
*picture of ASML laboratory*

> Fun fact: there are no major competitors to ASML today. It took them 30 years to reach the stage they're in today. They've integrated thousands of suppliers together to build a generator that fires 50,000 droplets per second. They also own the major company Cymer that makes these EUV sources. These light sources are so short (as short as 13.5nm) that there is no natural source for it.

OK, now we know what a transistor is and how it's made. Now we can understand GPUs (graphical processing units). A single GPU contains billions of transistors, which are packed onto a die (a raw block of silicon) manufactured with a 3nm fabrication technology. Don't worry, we'll get into these different types of fabrication technologies in just a bit. 

## Die Shrinks: Going from 10,000nm -> 2nm in 5 decades

| Terminology    | Definition                                                                   |
| -------------- | ---------------------------------------------------------------------------- |
| Die            | Microchip cut out from a silicon wafer                                       |
| Die shrink     | Manufacturing technique that redraws the same circuit design smaller         |
| Microprocessor | Type of chip that is used in the CPU (central processing unit) of a computer |
Let's take a quick history detour of die shrinks. Die shrinks mean that we're redrawing the same circuit design smaller and smaller, shrinking it!

**Why do we need it to be smaller?** This allows manufacturers (like TSMC) to produce a significantly higher number of dies on a single piece of silicon, lowering the cost it takes. There's also an additional side benefit, that chips will use less power and generate less heat, allowing for higher clock speeds and performance.

**What's the history of these dies and die shrinks?** Well, it's complicated, but I'll try to do a quick tour in a couple of paragraphs. The first microprocessor was launched in 1971, at a 10,000nm (or 10 micrometers) line width. This means that the gate length (distance between drain and source electrodes) was at 10,000 nm. This means that electrons have to travel across this 10,000nm whenever a transistor switches on. A shorter gate = faster speed.

At this point of time (1971), this chip, the Intel 4004, was designed for a Japanese calculator company, Busicom. However, once Intel realized that this was much more useful at mass market, Intel repurchased the marketing and technology rights from Busicom. A few years earlier, in 1965, Gordon Moore made the observation you've heard of as *Moore's Law*: that the number of transistors on a chip would keep doubling at low cost — originally about every year, which he revised in 1975 to roughly every two years.

Over the next few decades, we went from 600nm -> 250nm -> 180nm -> 130nm -> 45nm. In the early 2000s, manufacturers hit a wall. There was no way to take the next jump and shorten gate length. However, a breakthrough from a TSMC engineer named [Burn-Jeng Lin](https://english.cw.com.tw/article/article.action?id=3720) thought of adding water between the lens and the wafer. This was a huge bet by ASML in 2003-04, which was at that time a smaller European challenger behind Nikon and Canon. They went all in on immersion and won. 

Nikon and Canon stuck to their guns on 157nm dry lithography, but by then it was too late. ASML had a huge headstart. Canon essentially exited leading-edge lithography, and while Nikon did eventually build immersion tools, it never recovered the lead, and later sat out EUV entirely.

Today, your iPhones run on 3nm chips. GPUs today all use TSMC's 5/4/3 nm variants. We're currently at 2nm, and there's targets to hit 1.6nm (TSMC's "A16") around late 2026–2027, with 1.4nm later and true 1nm not expected until the back half of the decade. 

Unfortunately, one sad fact is that these numbers no longer mean gate lengths. They’re used more for marketing. If anything they refer to transistor density over the previous node, measuring MTr/mm^2 (millions of transistors on a mm^2).

![Screenshot 2026-05-30 at 8.04.24 AM](/images/unpacking-ai-hardware/Screenshot%202026-05-30%20at%208.04.24%20AM.png)
*Five decades of shrinking dies.*
## The Shift from CPU to GPU

Let's talk a bit about how GPUs got famous in the first place. It all started with the CPU.

CPUs have been in place since 1971, since the first microprocessor. However, when games like Quake were introduced in 1990s, computers were lagging pretty badly trying to render graphics. I remember my own computer grinding to a halt whenever a game was played. 

GPUs were designed exactly by NVIDIA to solve this. Instead of having a few sophisticated cores, you'd have thousands of dumb cores. Each individual core is super weak, but when combined together in a GPU, the throughput is gigantic. These were great for games, since rendering a 4K image meant computing colors of 8M pixels independently.

In 2006, Jensen Huang, CEO of NVIDIA made a huge bet. That Moore's Law is slowing. Single-threaded CPU performance was not optimized for the long run. He wanted to build a programming platform for scientific computing on graphics cards. This bet was targeted at scientists who wanted access to supercomputers. At that point of time, these supercomputers were multi-million dollar machines only owned by government labs and a few corporations.

This bet was called **CUDA** (Compute Unified Device Architecture). This allowed the CPU to offload parallelized computing tasks from the CPU to the GPU. This ended up being their moat. The ecosystem (PyTorch, TensorFlow, which we will cover in later chapters) ended up being built CUDA-first. Silicon and networking was also specialized around this architecture.

The first hint of AI leveraging GPUs came about in 2012. Three University of Toronto researchers, Alex K., Ilya S. and Geoffrey H. submitted a neural network called **[AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)**. This was trained on two NVIDIA GTX 580 gaming GPUs in Alex's bedroom. This proved that GPUs were feasible to train deep neural networks for the first time. These neural networks, which were mostly based on matrix multiplication was able to be achieved in someone's bedroom. If the same neural net were to be trained on a CPU, it would have taken centuries.

![Screenshot 2026-05-27 at 10.43.29 AM](/images/unpacking-ai-hardware/Screenshot%202026-05-27%20at%2010.43.29%20AM.png)
*How a GPU is structured. You can see that there are many more cores in a GPU! But at the same time, there's less "Control" and "Cache" layers, these are shared. Taken from the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/11.2.0/pdf/CUDA_C_Programming_Guide.pdf).*
## Structure of an NVIDIA GPU

| Acronym | Definition                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| SM      | Streaming Multiprocessor - self-contained core                                                             |
| GPU     | Graphical Processing Unit - explained in prior sections                                                    |
| GPC     | Graphical Processing Cluster - group of streaming multiprocessors                                          |
| MIG     | Multi-Instance GPU - sliced up into isolated logical GPUs to let providers run multiple tenants            |
| HBM     | High Bandwidth Memory - stacked DRAM chips sitting on the same package as the GPU                          |
| DRAM    | Dynamic random-access memory - one transistor + one capacitor per bit. cheap but needs constant refreshing |
| SRAM    | Static random-access memory - faster memory with six transistors, fast but expensive.                      |
| CUDA    | NVIDIA's proprietary programming model on their GPU architecture                                           |
| TSV     | Through-silicon via (tiny vertical wires) that connect HBM DRAM stacks.                                    |
Now let's take a look at a GPU. 

*Fair warning: it starts to get very technical from here on out. I will try my best to break them down.*

The first is an overall view of the Blackwell GPUs (launched Q4 2024). This is not the latest chip architecture: Rubin R100 was recently announced and plans to ship in a few months. I could not seem to find a good infographic on R100, so let me know if you do!

Nevertheless, this does give us a good sense of how it works end-to-end.
![Pasted image 20260527140656](/images/unpacking-ai-hardware/Pasted%20image%2020260527140656.png)

*Cleaned up version of the image from [Blackwell's overview](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/).*

Here in Blackwell, we have **2 dies** that have been welded together with a custom interconnect called NV-HBI. An interconnect is basically a physical wire linking two things together. In this case, NV-HBI is an ultra-low latency, proprietary die-to-die interconnect that powers 10 TB/s.

Now, let's think of each die as a *city*. Each city contains:

**🟩 4 GPCs (Graphics Processing Clusters), each holding 20 SMs**
A GPC is a _district_ within the city. Think of it as a district housing a group of factories that share some local infrastructure. Within a GPC, there are 20 SMs, which means that there are 80 SMs per die, and 160 SMs total across both dies. That's a lot of compute power!

**📋 GigaThread Engine + MIG Control**
The city-level dispatcher. Its job is simple: to receive work from the CPU (through the PCIe Gen 6) and farm it out to the GPCs. It also has an accompanying control portion, where the "MIG" part stands for _Multi-Instance GPU_. This lets the chip be sliced into up to 7 logical GPUs. Each of these GPUs looks isolated to different tenants, which is important for cloud providers running multiple customers on one physical chip.

**🟦 L2 Cache**
We have 50MB of shared [SRAM](https://en.wikipedia.org/wiki/Static_random-access_memory) (static random-access memory) in each L2 cache. Remember, SRAM is expensive, high-access RAM. Given this dual-die architecture, any SM can read what the other SM wrote. L0 and L1 caches are *within* the SM, which we'll cover in the next section.

**🟦 8 HBM3E stacks (288 GB total, up to 8 TB/s)**
Surrounding the dies, we have these High-Bandwidth Memory (HBM) stacks. In order to build them, we stack DRAM (dynamic random-access memory), with 4 stacks per die. 

These DRAMs are connected by Through-Silicon Vias (TSVs), aka microscopic vertical wires drilled through the silicon, connected by an interposer which is a thin slab of silicon that sits beneath the GPU die and HBM stacks.

The 'E' in HBM3E stands for extended – a refresh of the HBM3. Only three companies in the world make the HBM: SK Hynix (~55% share), Micron and Samsung (~20% each). HBM4 is already being sampled and ramping up for NVIDIA's Rubin.

> **On SRAM vs DRAM:** SRAM uses 6 transistors per bit and holds its value as long as it's powered on. It's expensive since it's bulky, but much faster. DRAM uses 1 transistor + 1 capacitor, but the whole array refreshes thousands of times per second. As such, SRAM lives on the GPU die which is closer to the GPC, and faster but more expensive. On the other hand, DRAM lives off the die. Both are however volatile, and gets lost once power gets cut. We'll dive into this deeper during the memory-compute wall.

Last but not least, we have the I/O paths surrounding the image:
- **NVLink v5** (1.8 TB/s) → connects to other GPUs via NVSwitch
- **PCIe Gen 6** (256 GB/s) → connects to the host system
- **NVLink-C2C** (900 GB/s) → connects to a paired CPU coherently (the Grace+Blackwell "superchip" combo). These are interconnects which we'll learn about shortly.

## Within a Streaming Multiprocessor
Now within a Graphical Processing Cluster (GPC), there are 20 Streaming Multiprocessors (SM).

What does each SM contain? Let's take a look.
![Pasted image 20260527140145](/images/unpacking-ai-hardware/Pasted%20image%2020260527140145.png)
Each SM is split into 4 partitions, which are all done in parallel. Remember we mentioned that Blackwell Ultra has 160 of these SMs, so this means 640 partitions in total.

**📋 L0 Instruction Cache**
This is a super fast cache that sits next to the Warp Scheduler. When the warp scheduler needs the next instruction (aka what to do, be it a matmul or a different operation), it pulls from this.

**📋 L1 Instruction Cache (top strip)**
This contains the recent instructions that are shared across all 4 partitions. If there's an L0 miss, it hits the L1 Cache.

**📣 Warp Scheduler (32 threads/clock)**
This is the shift manager, where upon every clock cycle, it picks a warp (32 threads) and issues an instruction. Remember, this is a GPU here, so all 32 threads in the warp execute the same instruction simultaneously on different data. 

**📤 Dispatch Unit**
The dispatch unit next to the Warp Scheduler helps to find the right execution unit. Now, there's going to be different execution units for different purposes. A CUDA core is used for regular multiplication (e.g. 3 x 2), a Tensor Core is used for a matrix multiplication, and an SFU is used for transcendental functions (exponential, logarithmic, trigonometric, etc.).

**🟦 Register File (64KB)**
This is the fastest possible storage. Registers are sitting physically adjacent to the execution units. And working values are kept here. This is used especially to tune performance. Keep note of this, as we will return to this when we talk about the memory-compute bottleneck.

**🟪 CUDA Cores**
This is where work happens. Each partition contains an assortment of execution units here. FP32 refers to 32-bit floating point math, INT32 is for integer math, and FP64 is for scientific computing. 

> A key point to note here: every time a clock cycle happens, a CUDA core does one multiply-add. A fun point, CUDA cores were introduced in 2006 in the GeForce 8800 GTX, as a shader.

**🟪 Tensor Cores**
Remember how AlexNet was introduced in 2012? Everyone wanted to train neural networks after that happened. However, CUDA cores were limited to one multiply-add per clock. This is because they were for scalar arithmetic. Given that neural networks are matrix multiplications, there needed to be a different type of cores.

At GTC 2017 (Nvidia's conference), the Volta V100 was launched, and the Tensor Core was introduced. It specialized in 4x4 matrix multiplication (matmul). That's 64 multiply-adds per clock, already a 64x improvement! With each V100 SM handling 8 Tensor Cores, that was an insane amount of matmul work it could perform, roughly 125 TFLOPs.

By the time Blackwell arrived, new floating points FP6 and FP4 were supported with adaptive precision selection. In layman terms, it was extremely powerful at 15 petaflops (10^15) per second. This means 15,000,000,000,000 floating operations per seconds.

Again, this is all NVIDIA. We'll take a look at other architectures like Google down the road. CUDA cores and Tensor Cores don't exist outside of NVIDIA.

Gleaning across the next two:
**🟧 SFU** - These Special Function Units help to take care of transcendental operations (sine, cosine, exponential, logarithm, softmax, math functions that cannot be represented by basic algebra).
🟥 **LD/ST Units**: Move data between registers and larger memory tiers.

**🟦 Tensor Memory (TMEM) — 256 KB**
New in Blackwell. A dedicated SRAM pool that is reserved exclusively for tensor (multi-dimensional scalar/vectors) operations. Used to stash work in progress instead of forcing data back to L1 Cache.

**🟦 L1 Data Cache / Shared Memory — 256 KB (configurable)**
Shared workshop SRAM that all 4 partitions can access.

## Blackwell by the numbers
When you add it all up, a single Blackwell Ultra SM contains:
- 4 partitions
- 128 CUDA cores (32 × 4 partitions)
- 64 INT32 units, 64 FP64 units
- 4 fifth-generation Tensor Cores (1 per partition)
- 256 KB total register file (64 KB × 4 partitions)
- 256 KB Tensor Memory (new)
- 256 KB L1 / shared memory
- 4 Texture units
- 4 Warp Schedulers, 4 Dispatch Units
- L0 instruction caches per partition, shared L1 instruction cache for the whole SM
- LD/ST units, SFUs across all partitions

Multiply by **160 SMs on the full chip** and you get the scale:
- ~20,480 FP32 CUDA cores total
- ~10,240 INT32, ~10,240 FP64
- **640 fifth-generation Tensor Cores**
- ~40 MB of TMEM across the chip (a new tier of on-die SRAM)
- ~40 MB of L1/shared memory
- ~100 MB of L2 cache (shared between the two dies, fully coherent)

Now that's a lot! No wonder it costs around $30-40k for each GPU.

## NVIDIA vs Google vs Others
Now let's take a break from numbers. Let's go back to history and understand who else is in this space.

There are three other major players. AMD, Google and Amazon.

AMD sells chips called Instinct. Google rents TPUs, or Tensor Processing Units. And Amazon also rents their chips called Trainium/Inferentia. 

There's also Groq and Cerebras which are newer companies, formed in 2016. You might have heard of Cerebras as the "first AI-era IPO", and Groq as a company started by a former TPU engineer that's been bought by NVIDIA in 2025. 

I won't cover them here in detail, but it's worth checking them out. Groq bets on inference using a new LPU (Language Processing Unit), and Cerebras is making one huge chip to reduce interconnect tax.

Let's put these four (NVIDIA, AMD, Google and Amazon) into a table.

|                   | NVIDIA GPUs                    | AMD Instinct GPU                                  | Google TPUs                                           | AWS Trainium / Inferentia                   |
| ----------------- | ------------------------------ | ------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------- |
| **What it is**    | Industry standard merchant GPU | Lower cost GPU but software is slowly catching up | Custom ASIC (application-specific integrated circuit) | Specialized chips for training / inference. |
| **Sold / Rented** | Sold                           | Sold                                              | Rented                                                | Rented                                      |
| **Market Share**  | 80%                            | 5-10%                                             | 6-8%                                                  | 2-3%                                        |
| **Software**      | CUDA - 15 year moat            | ROCm - open-source software                       | JAX + TensorFlow (we'll cover this in a future post)  | Neuron SDK                                  |
| **Customers**     | Almost everyone                | Microsoft and Meta, recently OpenAI               | Google Internal, Anthropic                            | Amazon, Anthropic                           |
What's interesting is that internal adoption - in the past Google and AWS used to run on NVIDIA GPUs within their clouds. But they're slowly starting to displace NVIDIA inside their own clouds. For instance, TPU is 75% of Google's Gemini and Trainium is 50% of AWS's Bedrock.

Anthropic is another interesting customer, because it's serving Claude across both Google and AWS clouds. And of course, as of three weeks ago, Anthropic announced a deal with SpaceX to use all the compute capacity at Colossus 1 in Memphis (300MW worth). This is spread across at least three silicon types now - AWS, Google Cloud and NVIDIA (the Colossus GPUs, from what was xAI).

## A Quick History: NVIDIA vs Google vs AWS

Most of the history revolves around NVIDIA and Google. NVIDIA starts at around 2006 with CUDA, and then Google starts their TPU journey around 2013, before bringing them public around 2018. 

> If there's just one takeaway from each of these two companies, it's this. Nvidia asks: How do I make threads more productive? Google asks: How do I keep the grid fed? How can I adjust my schedules?

#### NVIDIA
We'll talk about launches within NVIDIA, since each of these are important and have interesting bets. 

The codenames refer to GPU architecture codenames.
* **2006:** Launches *Tesla* with the CUDA architecture. This is the year where Jensen bet that parallel, individually weaker compute is going to be more important than large CPUs.
* **2010:** *Fermi*. This introduced a full IEEE-754 floating point, teaching computers how to do decimal/math operations. ECC memory was also a milestone here where this memory could detect single corrupt bits that could ruin the entire job.
* **2012**: *Kepler*. CUDA core counts jumped to 1500. This was the year AlexNet was trained on two GTX 580s as well.
* **2017**: *Volta* (V100). Each of these chips now have the first letter of their chip's name, as well as 100, which means the most performant chips.
* **2020**: *Ampere* (A100). 8-GPU boxes were added, with new number formats BF16 and TF32. Basically, the less number of bits, the faster the compute, but less precision. For neural networks, these were helpful.
* **2022**: *Hopper* (H100): Named after Grace Hopper, this was when the streaming multiprocessor turned into a dispatcher.
* **2024**: *Blackwell* (B200). These are dual dies that we talked about, and chips are now optimized for FP4 (4-bit floating point numerals). 
* **2026**: *Rubin* (R100) <-- We are now here! A new 3rd gen transformer engine was introduced recently.

#### Google TPUs
* **2013**: Internal TPUs get developed. Silicons are spent mostly on a systolic matrix engine.

Let's stop here. The architecture of a TPU is important and I want to cover it. You'll hear these a lot. The basis of the TPU's matmul engine is a giant systolic matrix engine called MXU. The reason it's called systolic is because it pumps data through the chip in rhythm (similar to the systole of a human heartbeat).

A visual is probably the best way to look at the difference between a GPU and TPU.
![Screenshot 2026-05-28 at 10.13.28 AM](/images/unpacking-ai-hardware/Screenshot%202026-05-28%20at%2010.13.28%20AM.png)
***CPU**: A few cores that handle one hard task each and stores results to memory.* 
***GPU**: Thousands of small cores that store results to memory.*
***TPU**: Accumulates partial sums to pass to the next grid. Never has to return to memory.*

Each of these have different execution models you'll hear about as well: CPU → SIMD (single instruction, multiple data), GPU → SIMT (single instruction, multiple threads), and TPU → Systolic .

Of course, this might seem great. **However, remember that we are only looking at one thing that TPUs excel at, matmul.** They've been heavily optimized for matmul. 

But these systolic arrays are weaker at branching heavy logic, sparse workloads, and irregular control flows when it comes to GPUs. For instance, softmax, a very important function in neural networks can run slower in a TPU than a GPU. As such, Google added other separate specialized hardware blocks called VPU (Vector Processing Unit), and Compilers to help with these other operations.

Alright, let's finish the history of TPU.
**2017:** [TPU v1 paper](https://arxiv.org/abs/1704.04760) goes public, and proved that GPU was not the only way for ML going forward.
**2018:** *v3*. MXU is scaled up with liquid cooling. Starts 
opening to Google Cloud customers.
**2021:** *v4*. SparseCore (specialized engine for data dependent embeddings) and Optical Circuit Switching also launches.
**2023:** *v5e* and *v5p* both get launched. v5p is especially interesting since trillion-parameter training is finally launched with a 3D torus shape.
**2024:** *Trillium* (v6e). 4x the multiply-accumulate count with a smaller HBM on each chip. The goal here was to optimize for inference throughput and cost efficiency with less dependencies on chip memory.
**2025**: *Ironwood* (v7). 64x more chips in one domain.
**2026**: *TPU 8t and 8i*. This time, it follows AWS's lead by separating training and inference chips, and leans on Broadcom and MediaTek to help implement these chips with TSMC.

#### AWS Trainium / Inferentia
In 2019, AWS realized that they were too dependent on NVIDIA, and for a hyperscaler their size, this wasn't the best position to be in.

Given that Google had proved that hyperscalers could build ASICs, they started to build their own chips, starting with Inferentia (for inference).

![Pasted image 20260529142157](/images/unpacking-ai-hardware/Pasted%20image%2020260529142157.png)

**2019**: *Inferentia* enters the market, marketed as "good enough for a cheaper price".
**2021**: *Trainium* adds a new training chip.
**2024**: *Trainium2* launches with 30-40% better price performance, with Anthropic as the main customer.
**2025-2026**: *Trainium3* ships and processes over 50% of Bedrock's token throughput. Provides up to 2.52 PetaFLOPs of FP8 compute.

#### Brief Segue into Floating Points & Numeral Formats

Traditional computing standardized around FP32 (32-bit floating point). This was great for scientific computing and simulations. However, over time, research showed that neural networks could tolerate lower precision. Throughput also scaled like crazy with the reduction of bits. For instance a H100 chip can deliver 67 TFLOPs at FP32 and 3958 TFLOPs at FP8.

This kicked off a round of increasingly specialized numerical formats:
- FP32 → traditional scientific computing
- FP16 → early deep learning acceleration
- BF16 → wider exponent range for more stable training
- TF32 → NVIDIA’s Tensor Core optimized training format
- FP8 / FP4 → ultra-low precision formats optimized for modern large-scale inference. 

![floating_point_format_evolution](/images/unpacking-ai-hardware/floating_point_format_evolution.svg)
*A brief visualization of floating point numeral formats*

We're currently at FP8/FP4 with NVIDIA's Rubin and Google's TPU, continuously learning how low number of bits an AI workload can tolerate. The art of performing this lossy compression is called quantization, where we are trying to reduce the memory footprint in order to increase inference speeds.
#### Interconnects
Before we jump into the final section, which I think is the most interesting problem of this piece, we need to talk about interconnects.

Why does it even matter? As GPU chips compute gradients during matmul operations, they need to share their gradients with each other. TPUs might have solved some of these problems with systolic operations, but that happens only within a chip. 

To share these gradients, they use **interconnects**.

NVIDIA has a full interconnect stack strategy. We talked about most of these in the prior section.
* **Within the chip:** NV-HBI (NVIDIA's High Bandwidth Interface) runs at 10 TB/s between the two dies.
* **Chip-to-chip:** NVLink 6 runs at 3.6TB/s for the Rubin series. Throughput roughly doubles every generation.
* **Rack-to-rack:** NVLink Switches which were introduced in Blackwell.
* **Server-to-server:** InfiniBand owns this layer. This is built by Mellanox, which was acquired by NVIDIA in 2020.

Whereas Google on the other hand takes a bit of a different approach (take note of their torus topology and OCS tech):
* **Within the chip:** No need for an interconnect
* **Chip-to-chip:** Inter-chip Interconnect (ICI): Runs at 9.6 Tbps. Uses a 3D torus topology which is a 3D lattice that loops back to itself. This is a lot cheaper at scale and given its twisted shape, every chip is connected to many others.
	* Optical Circuit Switching (OCS) is also a key innovation here that we mentioned earlier. At provision time, the interconnect topology can be rewired to match the workload.

It's also important to look at what else is happening across the landscape. Other interesting companies are:
* **Cerebras:** No interconnect. Don't even cut the chip wafer into multiple chips. Build one gigantic chip.
* **Groq:** No dynamic networking. All network traffic is statically scheduled by the cluster.

As well as new technologies that are being developed:
* **NVLink Fusion:** Allowing third-party chips to plug into NVLink. Developed by NVIDIA.
* **Ultra Ethernet:** Open standard built by AMD, Microsoft, Meta. This competes with InfiniBand and NVLink.
* **Optical I/O:** Photonic chiplets and interposers that route signals as light instead of copper traces. Today signals leave chips electrically and need to be converted to light and then back. If we remove that conversion, the energy savings is massive. Ayar Labs and Lightmatter are two companies that are tackling this space.

Now we'll talk about the biggest problem that GPUs have faced for awhile: the memory-bandwidth.

## The Biggest Issue: Memory Bandwidth

While GPU floating-point ops (aka compute) have been scaling exponentially every few years, memory bandwidth has not kept up. Memory Bandwidth = the speed at which data travels between the GPU chip and memory banks. This is now too slow. 

This leads to underutilized compute on the GPU (meaning that you're not getting the most of what you're paying for).
#### Rooflines and Arithmetic Intensity
Before we dive in further, we need to know what rooflines are. Roofline models help us understand whether an algorithm is memory bound or compute bound.

Let's take a fictitious example. Say your chip is a kitchen. Your chef is your compute (e.g. GPU cluster). It can perform 1000 TFLOP/s of knife work. The runner is your memory bandwidth, where it has 2TB/s of legs sprinting to the pantry (HBM) and back. 

*Is your kitchen limited by chef speed or runner speed?* To do that, we need to calculate the peak compute: 1000 TFLOP/s ÷ 2 TB/s = 500 operations per byte fetched. This means that recipes that do more than 500 things with each byte (e.g. ingredient) are bottlenecked by the chef. Less than that, and it's bottlenecked by the runner.

Here's a real example with a graph. A chip that can do **1000 TFLop/s** of compute and pull **2TB/s** from memory (bandwidth), the arithmetic intensity is calculated as 1,000 ÷ 2 = **500 FLOPs per byte**. 

This means that if your algorithm's Arithmetic Intensity is below 500, you're bandwidth bound (limited by how fast you can move tensors). If it's Arithmetic Intensity is above 500, you're compute bound. I'll refrain from turning Arithmetic Intensity into an acronym, well... because it's also AI.
![roofline_gemv_vs_gemm_with_numbers](/images/unpacking-ai-hardware/roofline_gemv_vs_gemm_with_numbers.svg)
Now we have two algorithms, Algo 1 and Algo 2.

**Algo 1 — matrix × one vector**
- Math done: 2 × 10,000 × 10,000 = 200 million FLOPs
- Data read: 10,000 × 10,000 × 2 bytes = 200 MB
- Arithmetic Intensity = 200M ÷ 200M = **1 FLOP/byte**

1 is way below the ridge of 500, so it's bandwidth-bound. Each number you load gets used exactly once, so there's no way to climb. Actual speed ≈ 1 × 2 TB/s = **2 TFLOP/s** — just 0.2% of the chip's peak. This means that the chip is mostly sitting idle waiting for memory. What a waste!

**Algo 2 — matrix × matrix**
- Math done: 2 × 10,000 × 10,000 × 10,000 = 2 trillion FLOPs
- Data read: 2 × 200 MB = 400 MB
- Arithmetic Intensity = 2T ÷ 400M = **5,000 FLOPs/byte**

Now, 5,000 is well past the ridge of 500, so it's compute-bound and runs at the peak compute **full 1,000 TFLOP/s**. It's the same matrix, but now every number you load gets reused thousands of times, so memory keeps up easily and the chip stays busy.
#### Solving Memory Bandwidth

All the above architecture that we studied help to address this memory bandwidth issue: large amounts of SRAM, L1-2 caches, quantization (remember the floating point stuff?), interconnects (NVLink, etc.), and moving memory as close to compute as possible. 

In particular, the most interesting solution here is optimizing speed through HBMs, or High-Bandwidth Memory. 

SK Hynix, a Korean company, conceptualized stacking DRAM dies vertically and connecting them together using TSVs (through-silicon vias). This allows enormous amounts of data to move in parallel. 

However, HBM3e is getting increasingly expensive, given that manufacturing and packaging difficulties increases with dies per HBM stack. 

[One paper](https://arxiv.org/abs/2601.05047), by Xiaoyu M. and David P. that is most interesting shares more about four new research areas to address these memory and interconnect issues. I'd actually advise checking out [the paper](https://arxiv.org/pdf/2601.05047) if you have time. 

They propose four different hardware shifts, and I'll try my best to dissect them, though the paper does a much better job:
1. **High-bandwidth Flash:** stack NAND flash instead of DRAM. With HBM, you can only stack so much DRAM before heat and cost gtes high. Flash is cheaper, denser and its capacity keeps doubling. However Flash has its own write-endurance limits, but is useful for data that's written rarely and read enormously (like model weights!)
2. **Processing-near memory (PNM):** For datacenter LLM inference, it seems that shards with PNM can be 1000x larger, which would allow these partitions to have a low communication overhead. With processing-in-memory (PIM), arithmetic units inside DRAM means (1) weak compute since these units are fabricated for DRAM and (2) creation of many shards which creates a lot of communication overhead.
3. **3D memory-logic stacking:** In 2D solutions, the HBM sits besides the processor. This means that the memory bandwidth is capped by how much edge the logic die has. As the area grows quadratically, the shoreline only grows linearly. With 3D stacking, it runs the entire 2D area of the die, which allows bandwidth to scale with area quadratically instead of with the parameter.

![Screenshot 2026-05-28 at 3.36.32 PM](/images/unpacking-ai-hardware/Screenshot%202026-05-28%20at%203.36.32%20PM.png)

Right: 3D stacking, taken from [the paper](https://arxiv.org/pdf/2601.05047).

![2d_shoreline_vs_3d_face_connection](/images/unpacking-ai-hardware/2d_shoreline_vs_3d_face_connection.svg)
4. **Low-latency interconnect.** Build topologies inspired by tori with high connectivity. Reduce work outside of the chip network and process within the network. Optimize chip design that land small packets directly into SRAM. Improve reliability through local standby spares and accepting good enough results. Again, more details in the paper that I can't cover in a short paragraph.

## Five things to remember

1. **It all comes back to one problem: Memory Bandwidth.** Compute got fast much faster than memory could feed it. Every acronym in this piece (HBM, SRAM, NVLink, quantization, 3D stacking) is an attempt to shrink the distance between the two.
2. **Three companies, three chokepoints.** Designers like NVIDIA hand a blueprint to TSMC (~70% of all chip fabrication), which can't print it without ASML's EUV machines (a literal monopoly). Each of these three companies form a strong dependency chain.
3. **"nm" is mostly marketing now.** We went from 10,000nm to 2nm, and the transistors got faster and denser the whole way. But shrinking the chip was never going to fix the memory gap. If anything it made all that compute harder to feed.
4. **NVIDIA and Google answer the same question differently.** NVIDIA asks how to make each thread more productive (CUDA, Tensor Cores). Google asks how to keep the whole grid fed (systolic arrays that never go back to memory). Both are interesting strategies that we'll see how it turns out over the next few years.
5. **The roofline tells you which side of the wall you're on.** In order to know if your algorithm is starved for memory, work out its arithmetic intensity: how many operations you do per byte loaded. You're either compute-bound or memory-bound.

That's it! You made it to the end of the first part of this series. Stay tuned for Part Two: **Data & Model Architecture.** Learn about how models are made of. We'll cover the paper that started it all ("Attention is All You Need"), plus talk about transformers and diffusion models. And of course, we'll cover how training data is prepared for these models (what sources? how is the data decontaminated and filtered?)

As I work on this series, I'd appreciate feedback and comments. What did you like? What would you like to see?

---

## References (TO CLEAN UP)

https://horace.io/brrr_intro.html
https://x.com/MainzOnX/article/2044462083010662771
https://jax-ml.github.io/scaling-book/
https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf