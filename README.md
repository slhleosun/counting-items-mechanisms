# Mechanistic Analysis of Counting in Llama-3-8B-Instruct
--- 
Author: [Lihao Sun](https://sites.google.com/uchicago.edu/lihao-sun) (University of Chicago)

This repository implements a pipeline for investigating which layer in large language models (LLMs) maintain a running count of category‐matching words when processing prompts such as:
> Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
> Type: fruit
> List: [dog, apple, cherry, bus, cat, grape, bowl]
> Answer: (

The experiments consist of bahavioral (prompt construction + open model evaluations) and mechanistic (linear probe + patching) parts. 

## Behavioral 
### Prompt Construction
> 📁Relevant Files: prompt_suite.ipynb
We create a synthetic dataset of prompts of the form: 
> Count the number of words in the following list that match the given type,
> and put the numerical answer in parentheses.
> Type: <CATEGORY>
List: [w₁, w₂, ..., w_N]
> Answer: (

where:
- <CATEGORY> is drawn uniformly from a predefined set of semantic categories (“fruit,” “animal,” “vehicle,” “instrument,” “furniture”).
- Each word list is of random length between 6 and 12 words, sampled (without replacement) from the union of all category pools. 
- We ensure at least one matching word per list, so that the correct answer is never zero.
- We compute the ground‐truth integer answer by counting how many list entries belong to the selected category.

### Benchmarking Models
> 📁Relevant Files: behavioral_eval.ipynb ; behavioral_results/
We evaluate a set of open‐weight LLMs on 5,000 prompts in a zero‐shot regime (no chain‐of‐thought steps, no appended reasoning instructions). Each model is asked exactly the prompt (ending in Answer: (), and we capture its next‐token generation(s) to extract the predicted integer. 
Models Evaluated:
> "llama3-8b-instr" : "meta-llama/Meta-Llama-3-8B-Instruct"
> "llama3-8b"       : "meta-llama/Meta-Llama-3-8B"
> "llama3-70b-instr": "meta-llama/Meta-Llama-3-70B-Instruct"
> "llama3-70b"      : "meta-llama/Meta-Llama-3-70B"
> "qwen2.5-3b-instr": "Qwen/Qwen2.5-3B-Instruct"
> "qwen2.5-3b"      : "Qwen/Qwen2.5-3B"

#### Behavioral Results 

## Mechanistic
### Methodology
We focus on the Llama 3-8B-Instruct model and use a causal-mediation (activation-patching) framework. The analysis proceeds in: 
- Hidden-State Caching and Linear Probing
- Activation Patching and Causal Evaluation

Each phase is explained below.
#### Hidden-State Caching and Linear Probing
In order to study only those cases where the model demonstrably formed a valid internal count, we filter the predictions file to keep the first N examples (here, N = 200) where pred == gold. This yields a list of prompts for which the model’s final answer exactly matches the ground-truth count. We save these filtered lines to correct_predictions.jsonl and also store them in memory for downstream analysis.

- Hidden State Collection: Extract activations from all 32 layers while processing counting prompts
Linear Probe Training: Train Ridge regression probes at each layer to predict running counts from hidden states
- Probe Evaluation: Use R² scores to identify which layers best encode counting information




### Results 
Linear Probe Performance Across Layers
