import os
import json
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict
from together import Together
import pandas as pd
from pathlib import Path
from datasets import load_dataset


@dataclass
class EvaluationResult:
    """Stores the evaluation result for a single metric"""
    metric: str
    score: float  # 0.0 to 1.0
    reasoning: str
    raw_response: str


@dataclass
class OverallEvaluation:
    """Stores all evaluation results for a prompt-response pair"""
    prompt: str
    response: str
    context: Optional[str]
    correctness: EvaluationResult
    faithfulness: Optional[EvaluationResult]
    relevance: EvaluationResult
    helpfulness: EvaluationResult
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "context": self.context,
            "correctness": asdict(self.correctness),
            "faithfulness": asdict(self.faithfulness) if self.faithfulness else None,
            "relevance": asdict(self.relevance),
            "helpfulness": asdict(self.helpfulness)
        }
    
    def get_average_score(self) -> float:
        """Calculate average score across all evaluated metrics"""
        scores = [
            self.correctness.score,
            self.relevance.score,
            self.helpfulness.score
        ]
        if self.faithfulness:
            scores.append(self.faithfulness.score)
        return sum(scores) / len(scores)


class LLMJudge:
    """
    LLM-as-a-Judge evaluator using Together AI's GPT-OSS-20B model.
    Provides reference-free evaluation on multiple quality metrics.
    """
    
    # Evaluation prompts for each metric
    CORRECTNESS_PROMPT = """You are an expert evaluator assessing the correctness of an AI assistant's response.

Evaluate the following response based on how well it answers the user's question/prompt. Consider:
- Does the response directly address what was asked?
- Are the facts and information accurate?
- Is the reasoning sound and logical?
- Are there any errors or misleading information?

User Prompt:
{prompt}

AI Response:
{response}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<your detailed reasoning for the score>"
}}

Score Guidelines:
- 1.0: Perfect, accurate, complete answer
- 0.7-0.9: Good answer with minor issues
- 0.4-0.6: Partially correct with notable problems
- 0.1-0.3: Mostly incorrect or unhelpful
- 0.0: Completely wrong or irrelevant

Respond ONLY with the JSON object, no additional text."""

    FAITHFULNESS_PROMPT = """You are an expert evaluator assessing the faithfulness of an AI assistant's response to provided context.

Evaluate how well the response is grounded in and supported by the given context. Consider:
- Are all claims in the response supported by the context?
- Does the response add information not present in the context?
- Are there any contradictions with the context?
- Does the response avoid hallucinations?

Context:
{context}

User Prompt:
{prompt}

AI Response:
{response}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<your detailed reasoning for the score>"
}}

Score Guidelines:
- 1.0: All claims fully supported by context, no hallucinations
- 0.7-0.9: Mostly faithful with minor unsupported details
- 0.4-0.6: Mix of supported and unsupported claims
- 0.1-0.3: Many claims not supported by context
- 0.0: Completely unfaithful, contradicts context

Respond ONLY with the JSON object, no additional text."""

    RELEVANCE_PROMPT = """You are an expert evaluator assessing the relevance of an AI assistant's response.

Evaluate how relevant and on-topic the response is to the user's prompt. Consider:
- Does the response stay focused on the question asked?
- Is all information provided pertinent to the query?
- Does it avoid tangential or off-topic content?
- Does it address the core intent of the question?

User Prompt:
{prompt}

AI Response:
{response}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<your detailed reasoning for the score>"
}}

Score Guidelines:
- 1.0: Perfectly relevant, all content directly addresses the prompt
- 0.7-0.9: Mostly relevant with minor tangents
- 0.4-0.6: Partially relevant, some off-topic content
- 0.1-0.3: Mostly irrelevant content
- 0.0: Completely irrelevant to the prompt

Respond ONLY with the JSON object, no additional text."""

    HELPFULNESS_PROMPT = """You are an expert evaluator assessing the helpfulness of an AI assistant's response.

Evaluate how helpful and actionable the response is for the user. Consider:
- Does it provide useful, practical information?
- Is the level of detail appropriate?
- Would this help the user accomplish their goal?
- Is the response clear and well-structured?
- Does it anticipate follow-up needs?

User Prompt:
{prompt}

AI Response:
{response}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<your detailed reasoning for the score>"
}}

Score Guidelines:
- 1.0: Extremely helpful, comprehensive, actionable
- 0.7-0.9: Very helpful with good detail
- 0.4-0.6: Somewhat helpful but lacking depth or clarity
- 0.1-0.3: Minimally helpful
- 0.0: Not helpful at all

Respond ONLY with the JSON object, no additional text."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b"):
        """
        Initialize the LLM Judge evaluator.
        
        Args:
            api_key: Together AI API key (defaults to TOGETHER_API_KEY env var)
            model: Model to use for evaluation (default: openai/gpt-oss-20b)
        """
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together AI API key must be provided or set in TOGETHER_API_KEY environment variable")
        
        self.model = model
        self.client = Together(api_key=self.api_key)
    
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Make a call to the Together AI API.
        
        Args:
            prompt: The evaluation prompt
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            The model's response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error calling Together AI API: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON from model response, handling markdown code blocks.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")
    
    def evaluate_correctness(self, prompt: str, response: str) -> EvaluationResult:
        """
        Evaluate the correctness of a response.
        
        Args:
            prompt: The original user prompt
            response: The AI's response to evaluate
            
        Returns:
            EvaluationResult with correctness score and reasoning
        """
        eval_prompt = self.CORRECTNESS_PROMPT.format(prompt=prompt, response=response)
        raw_response = self._call_llm(eval_prompt)
        parsed = self._parse_json_response(raw_response)
        
        return EvaluationResult(
            metric="correctness",
            score=float(parsed["score"]),
            reasoning=parsed["reasoning"],
            raw_response=raw_response
        )
    
    def evaluate_faithfulness(self, prompt: str, response: str, context: str) -> EvaluationResult:
        """
        Evaluate the faithfulness of a response to provided context.
        
        Args:
            prompt: The original user prompt
            response: The AI's response to evaluate
            context: The context/source material the response should be faithful to
            
        Returns:
            EvaluationResult with faithfulness score and reasoning
        """
        eval_prompt = self.FAITHFULNESS_PROMPT.format(
            prompt=prompt, 
            response=response, 
            context=context
        )
        raw_response = self._call_llm(eval_prompt)
        parsed = self._parse_json_response(raw_response)
        
        return EvaluationResult(
            metric="faithfulness",
            score=float(parsed["score"]),
            reasoning=parsed["reasoning"],
            raw_response=raw_response
        )
    
    def evaluate_relevance(self, prompt: str, response: str) -> EvaluationResult:
        """
        Evaluate the relevance of a response to the prompt.
        
        Args:
            prompt: The original user prompt
            response: The AI's response to evaluate
            
        Returns:
            EvaluationResult with relevance score and reasoning
        """
        eval_prompt = self.RELEVANCE_PROMPT.format(prompt=prompt, response=response)
        raw_response = self._call_llm(eval_prompt)
        parsed = self._parse_json_response(raw_response)
        
        return EvaluationResult(
            metric="relevance",
            score=float(parsed["score"]),
            reasoning=parsed["reasoning"],
            raw_response=raw_response
        )
    
    def evaluate_helpfulness(self, prompt: str, response: str) -> EvaluationResult:
        """
        Evaluate the helpfulness of a response.
        
        Args:
            prompt: The original user prompt
            response: The AI's response to evaluate
            
        Returns:
            EvaluationResult with helpfulness score and reasoning
        """
        eval_prompt = self.HELPFULNESS_PROMPT.format(prompt=prompt, response=response)
        raw_response = self._call_llm(eval_prompt)
        parsed = self._parse_json_response(raw_response)
        
        return EvaluationResult(
            metric="helpfulness",
            score=float(parsed["score"]),
            reasoning=parsed["reasoning"],
            raw_response=raw_response
        )
    
    def evaluate_all(
        self, 
        prompt: str, 
        response: str, 
        context: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> OverallEvaluation:
        """
        Evaluate a response on all available metrics.
        
        Args:
            prompt: The original user prompt
            response: The AI's response to evaluate
            context: Optional context for faithfulness evaluation
            metrics: List of metrics to evaluate. If None, evaluates all applicable metrics.
                    Options: ["correctness", "faithfulness", "relevance", "helpfulness"]
            
        Returns:
            OverallEvaluation containing all evaluation results
        """
        if metrics is None:
            metrics = ["correctness", "relevance", "helpfulness"]
            if context:
                metrics.append("faithfulness")
        
        results = {}
        
        if "correctness" in metrics:
            results["correctness"] = self.evaluate_correctness(prompt, response)
        
        if "faithfulness" in metrics:
            if not context:
                raise ValueError("Context is required for faithfulness evaluation")
            results["faithfulness"] = self.evaluate_faithfulness(prompt, response, context)
        
        if "relevance" in metrics:
            results["relevance"] = self.evaluate_relevance(prompt, response)
        
        if "helpfulness" in metrics:
            results["helpfulness"] = self.evaluate_helpfulness(prompt, response)
        
        return OverallEvaluation(
            prompt=prompt,
            response=response,
            context=context,
            correctness=results.get("correctness"),
            faithfulness=results.get("faithfulness"),
            relevance=results.get("relevance"),
            helpfulness=results.get("helpfulness")
        )
    
    def evaluate_batch(
        self,
        test_cases: List[Dict],
        context_key: str = "context",
        prompt_key: str = "prompt",
        response_key: str = "response",
        metrics: Optional[List[str]] = None
    ) -> List[OverallEvaluation]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of dictionaries containing test cases
            context_key: Key name for context in test case dict (default: "context")
            prompt_key: Key name for prompt in test case dict (default: "prompt")
            response_key: Key name for response in test case dict (default: "response")
            metrics: List of metrics to evaluate for each case
            
        Returns:
            List of OverallEvaluation objects
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}...")
            
            prompt = test_case[prompt_key]
            response = test_case[response_key]
            context = test_case.get(context_key)
            
            evaluation = self.evaluate_all(
                prompt=prompt,
                response=response,
                context=context,
                metrics=metrics
            )
            results.append(evaluation)
        
        return results
    
    def load_and_evaluate_from_jsonl(
        self,
        jsonl_path: str,
        id_key: str = "id",
        input_key: str = "input",
        output_key: str = "output",
        metrics: Optional[List[str]] = None,
        load_reference_data: bool = True
    ) -> pd.DataFrame:
        """
        Load a JSONL file, evaluate outputs using LLM-as-a-judge, and return results as DataFrame.
        Optionally loads category and topic information from the HuggingFace dataset.
        
        Args:
            jsonl_path: Path to JSONL file containing test cases
            id_key: Key name for ID in JSONL (default: "id")
            input_key: Key name for input/prompt in JSONL (default: "input")
            output_key: Key name for output/response in JSONL (default: "output")
            metrics: List of metrics to evaluate (default: all except faithfulness)
            load_reference_data: Whether to load category/topic from HF dataset (default: True)
            
        Returns:
            pandas DataFrame with evaluation results including scores, reasoning, and metadata
        """
        # Load JSONL file
        print(f"Loading JSONL file from {jsonl_path}...")
        test_cases = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
        
        print(f"Loaded {len(test_cases)} test cases")
        
        # Load reference dataset if requested
        reference_data = {}
        if load_reference_data:
            print("Loading reference dataset from HuggingFace...")
            try:
                hf_dataset = load_dataset("pt-tutorials-temp/llm-judge", split="train")
                # Create mapping from id to category and topic
                for item in hf_dataset:
                    reference_data[item['id']] = {
                        'instruction': item['instruction'],
                        'task': item['task'],
                        'topic': item['topic']
                    }
                print(f"Loaded reference data for {len(reference_data)} items")
            except Exception as e:
                print(f"Warning: Could not load reference dataset: {e}")
                load_reference_data = False
        
        # Evaluate all test cases
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nEvaluating test case {i+1}/{len(test_cases)}...")
            
            test_id = test_case.get(id_key)
            prompt = test_case.get(input_key, "")
            response = test_case.get(output_key, "")
            
            if not prompt or not response:
                print(f"  Skipping case {test_id}: missing input or output")
                continue
            
            # Run evaluation
            evaluation = self.evaluate_all(
                prompt=prompt,
                response=response,
                context=None,
                metrics=metrics or ["correctness", "relevance", "helpfulness"]
            )
            
            # Build result dictionary
            result = {
                'id': test_id,
                'input': prompt,
                'output': response,
                'correctness_score': evaluation.correctness.score if evaluation.correctness else None,
                'correctness_reasoning': evaluation.correctness.reasoning if evaluation.correctness else None,
                'relevance_score': evaluation.relevance.score if evaluation.relevance else None,
                'relevance_reasoning': evaluation.relevance.reasoning if evaluation.relevance else None,
                'helpfulness_score': evaluation.helpfulness.score if evaluation.helpfulness else None,
                'helpfulness_reasoning': evaluation.helpfulness.reasoning if evaluation.helpfulness else None,
                'average_score': evaluation.get_average_score()
            }
            
            # Add faithfulness if evaluated
            if evaluation.faithfulness:
                result['faithfulness_score'] = evaluation.faithfulness.score
                result['faithfulness_reasoning'] = evaluation.faithfulness.reasoning
            
            # Add reference data if available
            if load_reference_data and test_id in reference_data:
                ref = reference_data[test_id]
                result['reference_instruction'] = ref['instruction']
                result['task'] = ', '.join(ref['task']) if isinstance(ref['task'], list) else ref['task']
                result['topic'] = ', '.join(ref['topic']) if isinstance(ref['topic'], list) else ref['topic']
            
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"Evaluation complete!")
        print(f"Total cases evaluated: {len(results)}")
        print(f"Average scores:")
        print(f"  Correctness: {df['correctness_score'].mean():.3f}")
        print(f"  Relevance: {df['relevance_score'].mean():.3f}")
        print(f"  Helpfulness: {df['helpfulness_score'].mean():.3f}")
        print(f"  Overall: {df['average_score'].mean():.3f}")
        if load_reference_data:
            print(f"\nReference metadata loaded for {df['task'].notna().sum()} cases")
        print(f"{'='*80}\n")
        
        return df
    
    def save_results(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: Literal['csv', 'json', 'excel'] = 'csv'
    ):
        """
        Save evaluation results to file.
        
        Args:
            df: DataFrame with evaluation results
            output_path: Path to save the results
            format: Output format ('csv', 'json', or 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to {output_path}")
