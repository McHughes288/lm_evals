# LM Evals

This repository contains code for evaluating AI language models based on their preference for answers that sound nice to humans but are actually incorrect. It involves generating a set of questions, filtering them based on their relevance and correctness, and then testing various AI models against the generated test set to evaluate their performance.

It uses approaches for generating testsets with language models from the paper "Discovering Language Model Behaviors with Model-Written Evaluations".

## Files

1. `generate.py`: Generates a set of questions using a few-shot approach.
2. `discriminate.py`: Filters the generated questions based on their relevance and correctness.
3. `sort.py`: Sorts and deduplicates the questions and creates a final test set.
4. `score.py`: Evaluates various AI models against the generated test set and calculates their accuracy.
5. `utils.py`: Contains utility functions used by other scripts.

## How to run
Note: If you'd like to run in parallel for increased throughput, run the commands within bash loops in separate terminal panes or run in the background and direct the output logs accordingly.

### Generate questions

```bash
for i in $(seq 10); do
  python -m generate --number_to_generate 250 --out_path ./exp/250_generation_$i.json
done
```

### Discriminate questions

```bash
for i in $(seq 10); do
  python -m discriminate --in_path ./exp/250_generation_$i.json
done
```

### Sort and join questions to create the final test set

```bash
python -m sort --in_paths ./exp_discriminate/250_generation_1.json ./exp_discriminate/250_generation_10.json ./exp_discriminate/250_generation_2.json ./exp_discriminate/250_generation_3.json ./exp_discriminate/250_generation_4.json ./exp_discriminate/250_generation_5.json ./exp_discriminate/250_generation_6.json ./exp_discriminate/250_generation_7.json ./exp_discriminate/250_generation_8.json ./exp_discriminate/250_generation_9.json ./exp_discriminate/250_generation_10.json --out_path ./eval/final_testset.json
```

### Score models

```bash
for model in ada babbage curie davinci text-ada-001 text-babbage-001 text-curie-001 text-davinci-001; do
  python -m score --model $model --testset ./eval/final_testset.json
done
```

### Get results

```bash
grep '"accuracy":' exp_results/*.json
```

## Requirements

Python 3.x is required to run the scripts in this repository. Additionally, you'll need to install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Results

<img width="881" alt="Screenshot 2023-05-08 at 13 42 23" src="https://user-images.githubusercontent.com/25829615/236827027-06465739-d21a-4f1c-9634-288d60b72526.png">

The figure illustrates the percentage of responses that align with the intended behavior of the language model, which is to select a pleasant-sounding yet incorrect answer, across various OpenAI models: Ada, Babbage, Curie, Davinci, and human-feedback-based FeedME models. This data is based on a set of 1,000 questions generated within this repository. All models exhibit roughly a 50% likelihood of selecting appealing yet erroneous answers, and no inverse scaling is observed in relation to model size or the implementation of Reinforcement Learning from Human Feedback (RLHF). The initial hypothesis suggested that RLHF training would promote the selection of such answers, resulting in a higher percentage of the intended behavior for RLHF models (indicated by the red trace in the figure) compared to the base models. While this may still be a possibility, the generated test set must be further refined to demonstrate the anticipated inverse scaling effect.

## Known Limitations

Upon inspection of the final testset, the discriminator code tends to keep lots of examples around common movements that cause controversy such as flat earthers or anti-vaxxers.

An improvement would be to try and remove these completely because these questions are less about “nice-sounding” and relying on the opinions of certain groups of people. In order to do this, good next steps would be:

* Improve the discriminator code to use language model logits rather than asking for the model to give a confidence score in its answer.
* Spend more time doing prompt engineering to avoid questions that revolve around movements just as flat earthers.


## Contributing

Feel free to submit a pull request if you want to improve or add new features to this repository.
