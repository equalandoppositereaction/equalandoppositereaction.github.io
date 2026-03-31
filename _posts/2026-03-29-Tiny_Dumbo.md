---
title: Tiny Dumbo
date: 2026-03-29 23:29:23 +0530
categories: [LM]
tags: []     
description: >-
  A pre-trained extreamly small auto regressive model trained on 0.94B tokens of tiny stories
pin: true
---

![Dumbo](/assets/img/dumbo_mascot.png){: width="800" }
_Our beloved MASCOT!!!_


There was no use of Large Language Models involved during the creation of this project other than code reviews.

This report was took 4 hours (~12AM to 4AM) to make.

**ENJOY!**

---
## Table of Contents

1. [Architecture](#architecture)
2. [Hardware and Training](#hardware)
3. [Training Log](#log)
4. [Outcome](#outcome)
5. [Failed Runs and Major BUGS](#bugs)
6. [Learnings and Conclusion](#conclusion)

---

## Architecture {#architecture}

The architectural choices were simple, whatever was the best technique I had knowledge of were to be used.
It ended up being a llama2-ish model. 
A sentencepiece tokenizer was trained and used for this project as my tokenizer written in python was too slow.

**The diagram does not correctly render on mobile devices**

**Note:** This flowchart was PAINSTAKINGLY made by hand :)


```

                          ┌────────────────────────────────────────────────────────────┐
                          │                   ┌────────────────────┐                   │
                          │                   │       Output       │                   │
                          │                   │    Probabilities   │                   │
                          │                   └────────────────────┘                   │
                          │                              ▲                             │
                          │                   ┌──────────┴─────────┐                   │
                          │                   │       Softmax      │                   │
                          │                   └────────────────────┘                   │
                          │                              ▲                             │
                          │                   ┌──────────┴─────────┐                   │
                          │                   │       Linear       │                   │
                          │                   │ (Output Embedding) │                   │
                          │                   └────────────────────┘                   │
                          │                              ▲                             │
                          │                   ┌──────────┴─────────┐                   │
                          │                   │      RMS Norm      │                   │
                          │                   └────────────────────┘                   │
                          │                               ▲                            │
                          │                               │                            │
                          │        ┌──────────────────────│───────────────────┐        │
                          │        │     ┌────────────────┘                   │        │
                          │        │     │                                    │        │
                          │        │ ┌───┴────┐                               │        │
                          │        │ │  Add   │◄───────────┐                  │        │
                          │        │ └────────┘            │                  │        │
                          │        │     ▲                 │                  │        │
                          │        │     │                 │                  │        │
                          │        │     │   ┌─────────────┴──────────────┐   │        │
                          │        │     │   │                            │   │        │
                          │        │     │   │        Position Wise       │   │        │
                          │        │     │   │        Feed Forward        │   │        │
                          │        │     │   │         with SwiGLU        │   │        │
                          │        │     │   │                            │   │        │
                          │        │     │   └────────────────────────────┘   │        │
                          │        │     │                 ▲                  │        │
                          │        │     │                 │                  │        │
                          │        │     │         ┌───────┴──────┐           │        │
                          │        │     │         │   RMS Norm   │           │        │
                          │        │     │         └──────────────┘           │        │
                          │        │     │                 ▲                  │        │
                          │        │     │                 │                  │        │
                          │        │     │ ────────────────┘                  │   * 8  │
                          │        │     │                                    │        │
                          │        │ ┌────────┐                               │        │
                          │        │ │  Add   │◄───────────┐                  │        │
                          │        │ └────────┘            │                  │        │
                          │        │     ▲                 │                  │        │
                          │        │     │                 │                  │        │
                          │        │     │   ┌─────────────┴──────────────┐   │        │
                          │        │     │   │                            │   │        │
                          │        │     │   │    Masked Grouped Query    │   │        │
                          │        │     │   │         Attention          │   │        │
                          │        │     │   │         with RoPE          │   │        │
                          │        │     │   │                            │   │        │
                          │        │     │   └────────────────────────────┘   │        │
                          │        │     │                 ▲                  │        │
                          │        │     │                 │                  │        │
                          │        │     │         ┌───────┴──────┐           │        │
                          │        │     │         │   RMS Norm   │           │        │
                          │        │     │         └──────────────┘           │        │
                          │        │     │                 ▲                  │        │
                          │        │     │                 │                  │        │
                          │        │     │ ────────────────┘                  │        │
                          │        │     │                                    │        │
                          │        │     └──────────────┐                     │        │
                          │        └────────────────────┴─────────────────────┘        │
                          │                             ▲                              │
                          │                   ┌─────────┴──────────┐                   │
                          │                   │  Token Embedding   │                   │
                          │                   └────────────────────┘                   │
                          │                             ▲                              │
                          │                   ┌─────────┴──────────┐                   │
                          │                   │       Inputs       │                   │
                          │                   └────────────────────┘                   │
                          └────────────────────────────────────────────────────────────┘

```

### Model Config
The low number of layers probably causes most of the performence issue in the model despite of the simplicity of the task
the d_model/fcn_dim ratio was choosen 3.5 motivated by the qwen 3.5 models that prove this to be better for smaller models
against the more conventional 2.66 ratio (and also yields as multiple of 64).

```bash
MODEL_CFG = {
    'num_layers': 8,
    'vocab_size': 10240,  
    'd_model': 512,
    'fcn_dim': 1792,
    'num_heads': 8,
    'num_groups': 4,
    'device': 'cuda',
    'dtype': torch.bfloat16,
}
```

### Training Config
The batch size (128+64) was the best option after multiple trys for maximizing the GPU memory utilization (~92%) while being a multiple of 64 for maximizing the core performence.

AdamW was used as the optimizer as I hadn't studied about muon enough. A learning rate scheduler with the following config was used:

Warm-up (600 steps) -> Cosine annealing (timm alpha min) -> Post-annealing (after alpha min)

```bash
BATCH_SIZE = 192
CONTEXT_LENGTH = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 600 
MAX_STEPS = None
MIN_LR_RATIO = 0.01 
GRAD_CLIP_NORM = 1.0
BETAS = (0.9, 0.95)
```

## Hardware and Training {#hardware}
All of the training, inference and storage was done on Google Cloud and its Virtual Machines.

### Training the Tokenizer
The tokenizers were trained on a standard 32 thread 64GB RAM machine (RAM was not fully utilized and less would have worked).
The training took less than 10 min and tokenization took about 15 min.

### Training the Model
The model was trained on a virtual machine with Nvidia L4 GPU and 32 thread 48GB config. (12 thread and 16 RAM would have been enough).
There was no particular reason for using this config other than no other powerfull GPUs were available and I was unable to install the correct drivers in the RTX 6000.
It took about 7.2 hours to train the final model.
A lot of experimental or failed runs were done for upto 8.4 hours each.

### Cost
A total of ₹8,420 worth Google Cloud credits have been utilized with about 7k for VM instences and other for cloud storage.
It is worth noting the final model was not the only model trained in this amount as I also trained few models on FineWebEdu but they were too smal to output any comprehendable text.
(also, I over-slept during one of the runs and the VM kept running for about 6.5 extra hours)

## Training Log {#log}
I used wandb.ai's logger for logging during this project. It is free with institution IDs.

The legend names are formated as [no._of_blocks, fcn_dim, warmup_steps]

![Note that aqua has higher fcn_dim than pink still they achive identical loss, even at the micro scale!](/assets/pictures/tiny_dumbo/tiny_dumbo_loss.png){: width="800" }
_Cross Entropy Loss of per run._
Note that aqua has higher fcn_dim than pink still they achive identical loss, even at the micro scale!

![Note that so far aqua and pink has similar performance and learning rate per step, this will be relevent in the next part](/assets/pictures/tiny_dumbo/tiny_dumbo_lr.png){: width="800" }
_Learning rate throughout the runs._
Note that so far aqua and pink has similar performance and learning rate per step, this will be relevent in the next part

### A Cool Observation!

![GPU memory utilized by every run](/assets/pictures/tiny_dumbo/tiny_dumbo_gpu.png){: width="800" }
_GPU memory utilized by every run._


Note that the run in Grey had the highest allocation of memory, then Pink and Aqua had the least out of the three. That means the machine could spent more time in computation during Grey's run compared to Aqua as it has to fetch the data from the drive less often. This is also supported in next image!

![Disk IO during every run](/assets/pictures/tiny_dumbo/tiny_dumbo_disk.png){: width="800" }
_Disk IO during every run._

We know that disk I/O are extreamly slow, we also know that larger models have more things to compute and hence take more time to train.

Keeping these things in mind we can hypothise that Aqua should take more time to train given its high Disk I/O and larger size. This did not happen and by some devine intervention Aqua completed its training significantly earlier than pink! (you can see in the GPU memory graph that aqua finished before pink) Even on the exact same machine, data, learning rate etc!

## Outcome {#outcome}

The expected outcome was for the model to produce gramatically correct coherent output. However, (maybe because of its small size) the final outputs were not coherent (at all) but somewhat gramatically mediocure.

**Example:**

With prompt: "Once upon a"
```
Once upon a big dog with a tree. It was the dog. The dog saw the dog and a cat. The dog was He sad and sorry. After a while, the sun started to get a while too. He gave the dog his bird and his dad were sad. They wanted to about what the boy wanted. The boy thought for a moment. He ran to the dog, and said, "Tom,!" The boy saw Tim and his friends. They hugged the bird and felt sorry. They smiled and said, "Thank you to you and want to play with me?" The boy

```
It is hard to tell what its trying to say but we can kind of tell its a story about some dog named Tom and a boy and bird or something like that but the story does not go anywhere and has no structure. It ends as the max context length was reached.


With no prompt,
```
1. appreciated?" The boy was scared, but the boy didn't want to play with the car. They were happy that the friend had a great day.

2. challenge storm up wanted all around, but it was very special to be happy to have had time!
```

Note: <|beginoftext|> was never in the training dataset!

With prompt: "<|beginoftext|> Once upon a time, there lived a little girl named Lilly"
```
1. <|beginoftext|> Once upon a time, there lived a little girl named Lilly. She found Tom and Sue. He made a big plans of the tree. He had to go! The bird was so happy.
2. <|beginoftext|> Once upon a time, there lived a little girl named Lilly. She saw that. He decided a big smile, "I'm no longer scared, but you have to be friends?" Suddenly, the boat said, "You're welcome, we need to find it! You need to have a good way to go on an adventure." The next morning, the boy said, "You can't play in the park. It can have a lot of fun! The ball is very important to eat it." The little boy smiled, said, "It's!" The boy smiled. The girl said, "Hello so I want to stay here to see you."
```

## Failed Runs and Major BUGS {#bugs}

### I
The first few runs (on a different dataset) had absolute gibrish as outputs as I had used only 4 layers, 256 d_model and 2.66 as the d_model/fcn_dim ratio so the model size excluding embeddings was about 3-4M! (8-9M including the embeddings). It also had the bug described in section **II**.

### II
During the initial run, the tokenizer was trained with the format <|beginoftext|> - one row of data - <|endoftext|> (this was the pink run in the loss charts), during inference, weird unknown characters were displayed that were shown as "??". At first I thought this was due to some bug in the inference code but I could not point it out.

**Example outputs:**
```
temp-1 top-k 2
Once upon a time, there lived a little girl named Lilly. She ⁇  She ⁇  had ⁇  a ⁇  big ⁇  red ⁇  bird ⁇  who ⁇  loved ⁇  to ⁇  play ⁇  with ⁇ . ⁇  One ⁇  day ⁇ , ⁇  she ⁇  went ⁇  outside ⁇  to ⁇  play ⁇  with ⁇  her ⁇  friends ⁇ . ⁇  She ⁇  saw ⁇  a ⁇  big ⁇  dog ⁇  with ⁇  a ⁇  big ⁇  dog ⁇ . ⁇  The ⁇  dog ⁇  wanted ⁇  to ⁇  play ⁇  with ⁇  the ⁇  dog ⁇ . ⁇ 
 ⁇ 
 ⁇ But ⁇  then ⁇ , ⁇  she ⁇  saw ⁇  a ⁇  little ⁇  bird ⁇  with ⁇  a ⁇  big ⁇  dog ⁇ . ⁇  The ⁇  dog ⁇  was ⁇  very ⁇  sad ⁇ . ⁇  " ⁇ I ⁇  want ⁇  the ⁇  dog ⁇  to ⁇  go ⁇  home ⁇ !" ⁇ 
 ⁇ 
 ⁇ A ⁇ s ⁇  the ⁇  bird ⁇  saw ⁇  the ⁇  dog ⁇ , ⁇  it ⁇  was ⁇  too ⁇  big ⁇ . ⁇  It ⁇  had ⁇  a ⁇  lot ⁇  of ⁇  fun ⁇ . ⁇  The ⁇  dog ⁇  had ⁇  a ⁇  lot ⁇  of ⁇  fun ⁇  and ⁇  the ⁇  little ⁇  girl ⁇  would ⁇  go ⁇  outside ⁇ . ⁇  The ⁇  little ⁇  girl ⁇  had ⁇  a ⁇  lot ⁇  of ⁇  fun ⁇ . ⁇ 


temp 0.95 top k 50
Once upon a time, there lived a little girl named Lilly. She ⁇ ," ⁇  said ⁇  the ⁇  boat ⁇ . ⁇ 
 ⁇ 
 ⁇ He ⁇  was ⁇  very ⁇  sad ⁇ . ⁇  He ⁇  was ⁇  very ⁇  pretty ⁇ . ⁇  He ⁇  put ⁇  the ⁇  rock ⁇  in ⁇  his ⁇  head ⁇  and ⁇  started ⁇  to ⁇  feel ⁇  a ⁇  little ⁇  girl ⁇ . ⁇  He ⁇  had ⁇  never ⁇  seen ⁇  something ⁇  from ⁇  the ⁇  box ⁇  again ⁇ . ⁇  The ⁇  bird ⁇  was ⁇  so ⁇  proud ⁇  of ⁇  his ⁇  all ⁇ . ⁇ 
 ⁇ 
 ⁇ The ⁇  little ⁇  girl ⁇  was ⁇  so ⁇  excited ⁇  that ⁇  he ⁇  was ⁇  like ⁇  a ⁇  happy ⁇ , ⁇  they ⁇  were ⁇  having ⁇  fun ⁇ . ⁇  They ⁇  were ⁇  happy ⁇  to ⁇  be ⁇  scared ⁇  of ⁇  the ⁇  bird ⁇ , ⁇  so ⁇  he ⁇  had ⁇  to ⁇  fly ⁇  in ⁇  the ⁇  park ⁇ . ⁇  He ⁇  were ⁇  so ⁇  happy ⁇  to ⁇  see ⁇  that ⁇  the ⁇  little ⁇  girl ⁇  was ⁇  not ⁇  a ⁇  happy ⁇  good ⁇  friend ⁇ . ⁇ 


same  but with bos
<|beginoftext|> Once upon a time, there lived a little girl named Lilly. She ⁇  She ⁇  is ⁇  very ⁇  end ⁇  of ⁇  her ⁇  car ⁇  who ⁇  wanted ⁇  to ⁇  make ⁇  it ⁇ . ⁇ 
 ⁇ 
 ⁇ Alice ⁇  was ⁇  very ⁇  excited ⁇  and ⁇  she ⁇  loved ⁇  to ⁇  help ⁇  her ⁇  family ⁇ . ⁇  She ⁇  asked ⁇  her ⁇  friend ⁇ , ⁇  " ⁇ Don ⁇ 't ⁇  be ⁇  friends ⁇ . ⁇  We ⁇  thought ⁇  you ⁇  would ⁇  come ⁇  with ⁇  me ⁇ . ⁇  It ⁇ 's ⁇  always ⁇  found ⁇  it ⁇ , ⁇  Tom ⁇ ." ⁇ 
 ⁇ 
 ⁇ H ⁇ er ⁇  mom ⁇  said ⁇ , ⁇  " ⁇ I ⁇  love ⁇  you ⁇ , ⁇  mom ⁇ . ⁇  You ⁇  are ⁇  very ⁇  much ⁇ . ⁇  It ⁇ 's ⁇  so ⁇  shiny ⁇ . ⁇  I ⁇  love ⁇  it ⁇ ." ⁇  Sam ⁇  was ⁇  so ⁇  happy ⁇  to ⁇  see ⁇  the ⁇  girl ⁇ . ⁇  She ⁇  put ⁇  the ⁇  book ⁇  in ⁇  the ⁇  sun ⁇  and ⁇  started ⁇  to ⁇  play ⁇  with ⁇  it ⁇ . ⁇  She ⁇  started ⁇  to ⁇  come ⁇  away ⁇ . ⁇ 
 ⁇ 
 ⁇ Tom ⁇  and ⁇  her ⁇  mom ⁇  were ⁇  ready ⁇  to ⁇  play ⁇  together ⁇ . ⁇ 
```

Note that the outputs are similar to the once we got from the final run.

This bug almost made me quit this project. But I came back the next day and trained the tokenizer without the bos token and it was solved.

### III

I wanted to experiment with different warmup_steps, during this, it was observed that no warmup steps resulted in the model's loss plateauing (probably NaN loss?) as shown bellow by the yellow run.

![nan loss lr](/assets/pictures/tiny_dumbo/tiny_dumbo_nan_lr.png){: width="600" }
_In the other runs the learning rate start from 0 and then rises to lr, here the learning rate start from lr._

![nan loss](/assets/pictures/tiny_dumbo/tiny_dumbo_nan.png){: width="600" }
_This was larger than any of the above models with 16 layers and 1024 d_model._

The reason is, during the inisial phase, the weights are random so the gradients are huge, a large learning rate could add to the fuel and the steps would be soo big that they jump out of the 16 bit float value.

## Learnings and Conclusion {#conclusion}



#TODO