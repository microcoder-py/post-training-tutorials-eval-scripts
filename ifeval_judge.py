import subprocess
import os
from datasets import load_dataset
import json
import nltk

class IFEvalSetup:
    def __init__(
        self,
        dataset_name="google/IFEval",
        split="train",
        output_dataset_path="ifeval_dataset"
    ):
        self.original_dir = os.getcwd()
        self.repo_url = "https://github.com/oKatanaaa/ifeval.git"
        self.repo_name = "ifeval"

        self.dataset_name = dataset_name
        self.split = split
        self.output_dataset_path = output_dataset_path
        self.jsonl_dataset = "ifeval_json.jsonl"

        os.makedirs(self.output_dataset_path, exist_ok=True)

        self.install_ifeval()

    def install_ifeval(self):
        print("Installing necessary libraries for evaluation")
        try:
            subprocess.run(["git", "clone", self.repo_url], check=True)
            os.chdir(self.repo_name)
            subprocess.run(["pip", "install", "."], check=True)


        except Exception as e:
            if e.returncode == 128:
                print(f"IFEval already installed")
            else:
              print(f"Error installing IFEval")
        finally:
            os.chdir(self.original_dir)

        subprocess.run(["pip", "install", "--upgrade", "nltk"])
        nltk.download('punkt')
        nltk.download('punkt_tab')

    def download_dataset(self):
        print(f"Downloading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"Downloaded {len(dataset)} samples")

        with open(self.jsonl_dataset, "w", encoding="utf-8") as f:
            for row in dataset:
                cleaned_kwargs = []

                for kwargs in row['kwargs']:
                  cleaned = {k: v for k, v in kwargs.items() if v is not None}
                  cleaned_kwargs.append(cleaned)

                record = {
                    "key": row["key"],
                    "prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": cleaned_kwargs
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return dataset

    def run_tests(self, jsonl_inputs_path: str):
        if not os.path.exists(jsonl_inputs_path):
            raise FileNotFoundError(f"JSONL file not found at path: {jsonl_inputs_path}")

        from ifeval import Evaluator, InputExample, instruction_registry, read_input_examples, read_responses
        from ifeval import instruction_registry

        evaluator = Evaluator(instruction_registry)
        input_examples = read_input_examples(self.jsonl_dataset)
        responses = read_responses(jsonl_inputs_path)

        report, all_outputs = evaluator.evaluate(input_examples, responses)
        print("Strict prompt accuracy:", report["eval_results_strict"]["prompt_accuracy"])
        print("Loose prompt accuracy:", report["eval_results_loose"]["prompt_accuracy"])

        return report
