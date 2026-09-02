#!/usr/bin/env python3
import argparse

from verl_distill.tools.convert_dcp import convert_dcp_component


def main():
    parser = argparse.ArgumentParser(
        description="Convert one state from a legacy PyTorch distributed checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--state-name", default="teacher_discriminator_model")
    args = parser.parse_args()
    result = convert_dcp_component(
        args.checkpoint,
        args.output,
        state_name=args.state_name,
    )
    print(
        f"converted state={result['state_name']} tensors={result['tensor_count']} "
        f"output={result['output']}"
    )


if __name__ == "__main__":
    main()
