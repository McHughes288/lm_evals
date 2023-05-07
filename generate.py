import argparse


def parse_examples(examples_path):
    with open(examples_path, "r") as f:
        content = f.read()

    return content.split("===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        type=str,
        default="./eval/examples.txt",
        help="path to human written examples",
    )
    args = parser.parse_args()

    examples = parse_examples(args.examples)
    print(examples)


if __name__ == "__main__":
    main()
