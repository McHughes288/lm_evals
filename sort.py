import argparse
import os
from operator import itemgetter

from discriminate import swap_choices
from utils import load_json, save_json

DATETIME_FORMAT = "%Y-%m-%dT%H-%MZ"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_paths", nargs="+", help="Pass in files to operate on", required=True
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./eval/final_testset.json",
        help="path to filtered questions",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.out_path is None:
        name = os.path.basename(args.in_path)
        args.out_path = f"./exp_sorted/{name}"

    rated_questions = []
    for json_path in args.in_paths:
        rated_questions.extend(load_json(json_path))

    sorted_questions = sorted(rated_questions, key=itemgetter("score"), reverse=True)

    # deduplicate
    saved_questions = []
    sorted_questions_dedupe = []
    for question in sorted_questions:
        question_main = question["question"].strip().split("\n")[0].strip()
        if question_main not in saved_questions:
            saved_questions.append(question_main)
            sorted_questions_dedupe.append(question)
        else:
            print(f"Duplicated: {question_main}")

    questions = sorted_questions_dedupe[:500]

    balanced_questions = []
    for i, question in enumerate(questions):
        question["answer"] = "A"
        balanced_questions.append(question)

        # get balanced A and B choices
        swapped_question = {
            "question": swap_choices(question["question"]),
            "score": question["score"],
            "answer": "B",
        }
        balanced_questions.append(swapped_question)

    save_json(balanced_questions, args.out_path)


if __name__ == "__main__":
    main()
