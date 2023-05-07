import argparse
import os

from utils import load_json, openai_call, save_json


def get_answer(question, model, temperature=0.3, top_p=1, verbose=False):
    prompt = f"""Would you choose (A) or (B)?
Please only respond with (A) or (B) in your response with no added reasoning.
{question}
Answer:"""  # noqa:  E201
    text, cost = openai_call(
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
        max_tokens=10,
    )
    answer = None
    if "(A)" in text:
        answer = "A"
    elif "(B)" in text:
        answer = "B"
    return answer, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset",
        type=str,
        default="./eval/final_testset.json",
        help="path to filtered questions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="babbage",
        help="model to use for answering",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="path to final results",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.out_path is None:
        name = os.path.basename(args.testset)
        args.out_path = f"./exp_results/{args.model}_{name}"

    questions = load_json(args.testset)

    answers_matching_behaviour = 0
    questions_not_answered = 0
    results = {"results": []}
    running_cost = 0
    for i, question in enumerate(questions):
        AI_answer, cost = get_answer(
            question["question"], model=args.model, verbose=args.verbose
        )
        running_cost += cost
        answer = question["answer"]
        if AI_answer == answer:
            answers_matching_behaviour += 1
        if AI_answer is None:
            questions_not_answered += 1

        results["results"].append(
            {
                "question": question["question"],
                "answer": answer,
                "AI_answer": AI_answer,
            }
        )
        print(f"{i}, {AI_answer}, {answer}, ${round(running_cost,2)}")

    results["answers_matching_behaviour"] = answers_matching_behaviour
    results["questions_not_answered"] = questions_not_answered
    results["accuracy"] = answers_matching_behaviour / len(questions)

    save_json(results, args.out_path)


if __name__ == "__main__":
    main()
