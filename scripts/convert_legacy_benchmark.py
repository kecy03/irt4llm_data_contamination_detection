import argparse

from contamination_schema import convert_legacy_rows, infer_task_name, load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy few-shot contamination JSONL into the unified benchmark schema."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--source_task", type=str, default=None)
    parser.add_argument("--source_split", type=str, default="test")
    parser.add_argument("--contam_type", type=str, default="fewshot")
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    converted = convert_legacy_rows(
        rows,
        model_id=args.model_id,
        source_task=args.source_task or infer_task_name(args.input_path),
        source_split=args.source_split,
        contam_type=args.contam_type,
    )
    save_jsonl(args.output_path, converted)


if __name__ == "__main__":
    main()
