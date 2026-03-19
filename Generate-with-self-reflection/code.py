'''
    Goal: Create a model that outputs a response and a self-reflection on that response. 
    
    The self-reflection should include an evaluation of the response's quality and a 
    plan for how to improve it in the future.
'''

from dotenv import load_dotenv
from pydantic import BaseModel,Field
import litellm

load_dotenv()

class ReflectionOutput(BaseModel):
    original: str = Field(exclude=True)
    revised: str
    needed_revision: bool

def call_llm(prompt: str, system_prompt: dict = {}, format_model: type = None):
    messages = [{"role": "user", "content": prompt}]

    if system_prompt:
        messages.append(system_prompt)
        print("Adding system_prompt")

    return litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        response_format=format_model
    ).choices[0].message.content

def generate_with_self_correction(user_prompt):
    # 1. THE DRAFT
    draft = call_llm(user_prompt)
    
    # 2. THE CRITIQUE
    critique_prompt = f"Review this draft for errors: {draft}\nList any mistakes."
    critique = call_llm(critique_prompt, None, ReflectionOutput)

    reflection_output_obj = ReflectionOutput.model_validate_json(critique)
    
    # 3. THE REFINEMENT
    if reflection_output_obj.needed_revision:
        final_prompt = f"Original Prompt: {user_prompt}\nCritique: {reflection_output_obj.revised}\nRewrite perfectly."
        final_response = call_llm(final_prompt)
        return final_response
        
    return draft

if __name__ == "__main__":
    self_reflection_inducing_prompt = "Write a sentence about the Roman god Jupiter using exactly 10 words, where every word must start with the letter 'J' or 'S'."
    print(generate_with_self_correction(self_reflection_inducing_prompt))