'''
Create an LLM that can automatically evaluate the quality of annotations provided by human experts.
The LLM should be able to compare the annotations against a set of ground truth annotations and provide 
a score based on accuracy and hallucination. Additionally, the LLM should be able to provide reasoning
for the scores it assigns, and generate a report summarizing the evaluation results and providing 
recommendations for improving the annotation process. The evaluation should be done in parallel for
multiple annotations to optimize for time efficiency.
'''

from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from dotenv import load_dotenv
from csv import DictReader
from pathlib import Path
import litellm
import argparse
import csv
from typing import Optional, Union
import asyncio

load_dotenv() # Load OpenAI API key

'''
    Schema Definition
'''
class Customer(BaseModel):
    id: str
    name: str

class Project(BaseModel):
    id: str
    name: str

class CategoryEnum(str, Enum):
    FEDERAL = "Federal"
    COMMERCIAL = "Commercial"

class StatusEnum(str, Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"

class TaskAnnotation(BaseModel):
    id: str
    customer_id: str
    project_id: str
    category: CategoryEnum
    prompt: str
    annotation: str
    version: int
    task_status: StatusEnum

'''
    Load Mock Data
'''
def load_mock_data(type: type, file_path: Path):
    print("Loaded mock data for ", type.__name__, " from ", file_path.name)

    data = []
    with open(file_path, 'r') as file:
        reader = DictReader(file, escapechar='\\', skipinitialspace=True)
        
        for row in reader:
            try:
                if type == MetadataScore:
                    row['reasoning_score'] = ReasoningScore.model_validate_json(row['reasoning_score'])
                obj = type.model_validate(row)
                data.append(obj)
            except Exception as e:
                print(f"Validation error on row {row.get('id')}: {e}")
                continue

    return data

'''
    Prompt + Answer Pair Evaluation Loop
'''

class EvaluationScore(BaseModel):
    task_id: Optional[str] = Field(default=None, exclude=True) # Task ID field (hidden from LLM)
    reasoning: str = Field(description="Step-by-step justification based on the rubric.") # Forced chain-of-thought reasoning to ensure the model explains its scores.
    accuracy: int = Field(ge=1, le=10)
    hallucination: int = Field(ge=1, le=10)

async def process_single_task(t: TaskAnnotation, system_prompt: str) -> Optional[EvaluationScore]:
    """Helper to handle retries and keep context for a single task."""
    MAX_ATTEMPTS = 2
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Prompt: {t.prompt} | Annotation: {t.annotation}"}    
                ],
                response_format=EvaluationScore
            )
            
            content = response.choices[0].message.content
            score_obj = EvaluationScore.model_validate_json(content)
            
            # Context is preserved here!
            score_obj.task_id = t.id
            return score_obj
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for task {t.id}: {e}")
            if attempt == MAX_ATTEMPTS - 1:
                return None # Or handle failure as needed
            await asyncio.sleep(1) # Small backoff before retry

async def evaluate(tasks: list[TaskAnnotation]) -> list[EvaluationScore]:
    print(f"Running evaluations for {len(tasks)} TaskAnnotations in parallel...")

    system_prompt = f"""
        You are a expert Quality Assurance judge at Scale AI. Your goal is to evaluate the accuracy and hallucination of LLM annotations.

        ### ACCURACY RUBRIC
        - 10: Perfect. Factually correct, follows all instructions, and is concise.
        - 7-9: Minor issues. Correct info, but perhaps slightly wordy or formatting is off.
        - 4-6: Major issues. Contains significant factual errors or misses a core part of the prompt.
        - 1-3: Total failure. Completely incorrect or irrelevant.

        ### HALLUCINATION RUBRIC
        - 1: No Hallucination. Every claim is supported by the context or common knowledge.
        - 5: Partial Hallucination. Mixes real facts with 1-2 invented details.
        - 10: Severe Hallucination. The entire response is fabricated or "fictionalized."

        ### INSTRUCTIONS
        1. Analyze the Prompt and Annotation step-by-step.
        2. Provide a 'reasoning' string explaining which rubric criteria was met.
        3. Output the final scores based strictly on the rubric above.
    """

    # Create a list of worker coroutines
    coros = [process_single_task(t, system_prompt) for t in tasks]

    # asyncio.gather now returns the actual results directly
    results = await asyncio.gather(*coros)

    # Filter out any None results from permanent failures
    evaluation_scores = [r for r in results if r is not None]

    return evaluation_scores    

'''
    Caching Logic to Save Compute / Time
'''

def cache_scores(scores: list[Union[EvaluationScore, 'MetadataScore']], destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'w', newline='', encoding='utf-8') as f:
        # Use csv.DictWriter for safety with long strings/commas
        if isinstance(scores[0], MetadataScore):
            fieldnames = list(MetadataScore.model_fields.keys())
        else:
            fieldnames = list(EvaluationScore.model_fields.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for ev in scores:
            row_data = ev.model_dump()
            if isinstance(ev, MetadataScore):
                row_data['reasoning_score'] = ev.reasoning_score.model_dump_json()

            row_data['task_id'] = ev.task_id
            writer.writerow(row_data)

    print(f"Cached {len(scores)} scores to: {destination.name}")

'''
    System Evaluation Loop
'''

class ReasoningScore(BaseModel):
    task_id: Optional[str] = Field(default=None, exclude=True)
    reasoning: str # The reason why this score was assigned
    score: int = Field(ge=1, le=10) # The reasoning score for the LLM

def score_reasoning_with_llm(task_id: str, expected_reasoning: str, actual_reasoning: str) -> ReasoningScore:
    judge_system_prompt = """
    You are a Senior Quality Auditor at Scale AI, specialized in Meta-Evaluation. 
    Your task is to perform a high-fidelity comparison between a "Ground Truth Reasoning" (the gold standard) and a "Candidate Reasoning" produced by an internal scoring model.

    ### EVALUATION CRITERIA: REASONING ALIGNMENT
    You must evaluate how closely the Candidate's logic aligns with the Ground Truth. Do not just look for keyword matches; look for logical equivalence.

    ### THE RUBRIC
    - 10 (Perfect Alignment): The Candidate captures the exact logical path of the Ground Truth. It identifies the same nuances, errors, or strengths.
    - 7-9 (Minor Variance): The logic is sound and reaches the correct conclusion, but may miss one minor nuance or use slightly less precise terminology than the Ground Truth.
    - 4-6 (Logical Divergence): The Candidate misses a primary justification found in the Ground Truth, or provides reasoning that is technically "fine" but fails to address the specific rubric violation identified by the expert.
    - 1-3 (Critical Failure): The Candidate contradicts the Ground Truth, justifies a wrong score with false logic, or produces irrelevant "filler" text.

    ### TASK INSTRUCTIONS
    1. Compare the Candidate Reasoning against the Ground Truth Reasoning.
    2. Identify if the Candidate reached the same conclusion for the same reasons.
    3. Provide a brief justification for your score, highlighting exactly where the logic diverged (if at all).
    4. Output the final score as an integer between 1 and 10.
    """

    error_msg = None
    
    for _ in range(2):
        try:
            messages = [
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": f"Expected reasoning: {expected_reasoning} | Actual reasoning: {actual_reasoning}"}
            ]

            if error_msg is not None:
                messages.append({"role": "user", "content": f"Error during schema validation: {error_msg}"})

            response = litellm.completion(
                model="gpt-4o-mini",
                messages= messages,
                response_format=ReasoningScore,
            )

            content = response.choices[0].message.content

            reasoning_score = ReasoningScore.model_validate_json(content)

            reasoning_score.task_id = task_id

            return reasoning_score
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt to score reasoning failed: {error_msg}\n\nRetrying...")

    return ReasoningScore(task_id=task_id, reasoning=f"Audit failed: {error_msg}", score=1)

class MetadataScore(BaseModel):
    task_id: str
    delta_accuracy: float
    delta_hallucination: float
    reasoning_score: ReasoningScore

class EvalReport(BaseModel):
    final_percentage: float = Field(exclude=True) # The final score achieved in this eval as a percentage (X%)
    summary: str # A summarization of the evaluation
    recommendation: str # Recommendations for improving scoring and tuning the model

    def __repr__(self):
        return f"Final percentage: {self.final_percentage:.2f}%\n\nSummary: {self.summary}\n\nRecommendations: {self.recommendation}"

    def __str__(self):
        return self.__repr__()

def generate_report(metadata: list[MetadataScore]) -> EvalReport:
    system_prompt = """
    You are a Senior AI Strategist at Scale AI. Your task is to analyze a batch of Model Evaluation results and generate a "Model Tuning Strategy Report."

    ### INPUT DATA
    You will receive a list of metadata entries. Each entry contains:
    - Task ID
    - Accuracy/Hallucination Deltas (Difference between LLM Judge and Human Expert)
    - Reasoning Alignment Score (How well the LLM's logic matches the expert's logic)
    - Meta-Judge Critique (Detailed feedback on where the LLM's logic failed)

    ### REPORT STRUCTURE
    Your report must follow this exact structure:

    1. EXECUTIVE SUMMARY
    - Provide a high-level "Grade" for the current scoring model (e.g., "Production Ready," "Needs Prompt Tuning," or "Unreliable").
    - State the average logic alignment percentage.

    2. QUANTITATIVE ANALYSIS
    - Identify the percentage of tasks where the model was "Strict" (scored lower than human) vs. "Lenient" (scored higher than human).
    - Report the frequency of "Critical Divergence" (tasks where reasoning alignment is < 4).

    3. QUALITATIVE PATTERNS (The "Why")
    - Analyze the Meta-Judge Critiques to find common themes. (e.g., "The model consistently misses subtle hallucinations in code snippets" or "The model is over-penalizing polite refusals.")

    4. ACTIONABLE TUNING STEPS
    - Provide 3 specific recommendations to improve the primary scoring model. 
    - Examples: "Update the Hallucination Rubric to define X," "Add 3 few-shot examples of Y to the system prompt," or "Swap to a larger model (e.g., GPT-4o) for this specific category."

    ### TONE
    Professional, data-driven, and highly analytical. Avoid fluff. Focus on "Root Cause Analysis."
    """

    error_msg = None
    
    for _ in range(2):
        try:
            metadata_dicts = [m.model_dump() for m in metadata]
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(metadata_dicts)}
            ]

            if error_msg is not None:
                messages.append({"role": "user", "content": f"Error during schema validation: {error_msg}"})

            response = litellm.completion(
                model="gpt-4o-mini",
                messages= messages,
                response_format=EvalReport,
            )

            content = response.choices[0].message.content

            report = EvalReport.model_validate_json(content)

            return report
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt to score reasoning failed: {error_msg}\n\nRetrying...")

    return EvalReport(final_percentage=0.0, summary="Report generation failed!", recommendation="N/A")

def _run_scoring(actual: list[EvaluationScore], expected: list[EvaluationScore]) -> list[MetadataScore]:
    actual_scores_map = {ev.task_id: ev for ev in actual}

    processed_ct = 0

    metadata = [] # Stores scores earned for each ticket_id to help us improve

    for exp in expected:
        act = actual_scores_map.get(exp.task_id)

        if not act:
            print(f"Skipping {exp.task_id}: No evaluation data found.")
            continue

        delta_accuracy = exp.accuracy - act.accuracy
        delta_hallucination = exp.hallucination - act.hallucination

        reasoning_score = score_reasoning_with_llm(exp.task_id, exp.reasoning, act.reasoning)

        # Record the run
        metadata_score = MetadataScore.model_validate({
            "task_id": exp.task_id,
            "delta_accuracy": delta_accuracy,
            "delta_hallucination": delta_hallucination,
            "reasoning_score": reasoning_score
        })

        metadata.append(metadata_score)

        processed_ct += 1

    return metadata

async def evaluate_agent(evaluation_dir: Path, args: any) -> EvalReport:
    '''
        Evaluates the agent responses on ground truth TaskAnnotations.

        Returns the score as a percentage of the number of points gained compared to the number of points lost.

        Each correct answer for accuracy is +1
        Each incorrect answer for accuracy is 0
        Each correct answer for hallucination is +1
        Each incorrect answer for hallucination is 0
    '''

    print("Running ground truth evaluation...")

    eval_task_ann_file_path = evaluation_dir/"eval_task_annotation.csv"
    test_task_annotation_data = load_mock_data(TaskAnnotation, eval_task_ann_file_path)

    ground_truth_file_path = evaluation_dir/"ground_truth.csv"
    ground_truth_eval_scores = load_mock_data(EvaluationScore, ground_truth_file_path)

    cache_dir = evaluation_dir.parent/"cache"
    cached_ground_truth_scores_file_path = cache_dir/"cached_ground_truth_eval_scores.csv"
    if not args.load:
        actual_eval_scores = await evaluate(test_task_annotation_data)
        cache_scores(actual_eval_scores, cached_ground_truth_scores_file_path)
    else:
        actual_eval_scores = load_mock_data(EvaluationScore, cached_ground_truth_scores_file_path)

    cached_metadata_scores_file_path = cache_dir/"cached_metadata_scores.csv"
    if not args.load:
        metadata = _run_scoring(actual_eval_scores, ground_truth_eval_scores)
        cache_scores(metadata, cached_metadata_scores_file_path)
    else:
        metadata = load_mock_data(MetadataScore, cached_metadata_scores_file_path)
    
    score = 0
    for m in metadata:
        score += max(0, 1 - (abs(m.delta_accuracy) / 10))
        score += max(0, 1 - (abs(m.delta_hallucination) / 10))
        score += max(0,  (m.reasoning_score.score/ 10))

    total_possible_points = len(metadata) * 3
    final_percentage = (score / total_possible_points) * 100

    eval_report = generate_report(metadata)

    eval_report.final_percentage = final_percentage

    return eval_report

'''
    Main Orchestrator Loop
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation Evaluation CLI Tool")
    parser.add_argument("--cache", '-c', action="store_true",help="This argument determines whether to cache the results.")
    parser.add_argument("--load", '-l', action="store_true",help="This argument determines whether to load EvaluationScore objects from the cache.")
    parser.add_argument("--eval", '-e', action="store_true", help="This argument determines whether or not to evaluate the current LLM.")

    args = parser.parse_args()

    if args.load and args.cache:
        raise Exception("Warning: You are trying to load from cache AND cache results. Loading will take priority.")
    
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir/"data"
    mock_task_annotation_file_path = data_dir/"mock_task_annotation.csv"
    mock_customer_file_path = data_dir/"mock_customer.csv"
    mock_project_file_path = data_dir/"mock_project.csv"

    task_annotation_objs = load_mock_data(TaskAnnotation, mock_task_annotation_file_path)
    customer_objs = load_mock_data(Customer, mock_customer_file_path)
    project_objs = load_mock_data(Project, mock_project_file_path)

    assert len(task_annotation_objs) == 12
    assert len(project_objs) == 4
    assert len(customer_objs) == 5

    if args.eval:
        eval_data_dir = script_dir/"evaluation"
        report = asyncio.run(evaluate_agent(eval_data_dir, args))

        print("\n\n", "="*10, "REPORT", "="*10)
        print(report)
    else:
        # Run random evaluation logic
        if not args.load:
            evaluation_scores = asyncio.run(evaluate(task_annotation_objs))
        else:
            evalation_score_file_path = script_dir/"cache/evaluation_score_cache.csv"
            evaluation_scores = load_mock_data(EvaluationScore, evalation_score_file_path)

        if not args.load and args.cache:
            cache_dir = script_dir/"cache"
            cache_file = cache_dir/"evaluation_score_cache.csv"

            cache_scores(evaluation_scores, cache_file)