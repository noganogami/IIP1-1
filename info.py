import argparse

from torchinfo import summary

from networks import Baseline, CustomInception, ResInception


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Baseline")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_option()

    if args.model_name == "Baseline":
        model = Baseline(num_classes=10)
    elif args.model_name == "Inception":
        model = CustomInception(num_classes=10)
    elif args.model_name == "ResInception":
        model = ResInception(num_classes=10)
    else:
        raise NotImplementedError("The model has not implemented: " + args.model_name)

    summary(
        model=model,
        input_size=(args.batch_size, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params"],
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
