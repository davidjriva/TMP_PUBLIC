- Very python heavy
- Evaluate skills that resemble day to day work for applied AI engineer
- LLM saviness
- System setup + initial evaluation
- Revised system implementation + evaluation
- Agent workflow understanding
- Common SDK familiarity
- Overall LLM saviness
- Python coding abilities + libraries
- Documentation lookup skill
- Testing and debugging capabilities
- Google and Gemini are allowed for single-shot questions. You can’t just tell it to update code and fix things.
- Evaluating a base LLM on a certain type of task— load data, run inferences. Mocked / they provide keys in a notebook. They provide a lite LLM keys.

---

Applied AI Coding
(60 minutes)

Interview Overview
This interview will assess your ability to work with structured data, analyze system performance, and propose improvements. You’ll work through building an LLM-based agent, reviewing data, debugging issues, and iterating on solutions.

What to Expect

• Understanding the Task: Reviewing provided artifacts and system setup.
• Data Processing: Structuring and organizing code to process artifacts.
• Analysis & Problem Solving: Identifying issues and proposing solutions.

How to Prepare

• Python Proficiency: Be comfortable navigating and modifying code.
• Structured Thinking: Focus on breaking down problems logically.
• Data Evaluation: Be ready to assess and interpret results.
• No ML Experience Required: But familiarity with LLMs, LLM-based agent design, and Completion APIs will help.

What We’re Looking For

✔ Clear problem-solving approach
✔ Logical reasoning and analysis
✔ Ability to iterate and refine solutions
💡 This is not about getting the "perfect" answer—it’s about demonstrating structured thinking and adaptability.

---

Predicted Interview Workflow

1. Setup & Evaluation (Minutes 0-20)
   The Task: You'll likely be given a Jupyter notebook or a Python script with a few example inputs (JSON/CSV) and
   a baseline prompt.

Coding Action: You need to write a helper function to load this data and run a few inferences.

The Catch: The base model will probably perform poorly—it might hallucinate, fail to follow formatting,
or miss edge cases. You'll need to calculate a basic metric (e.g., Accuracy or a "Pass/Fail" check) and
state clearly why it's failing.

2. System Implementation & "Agent" Design (Minutes 20-45)
   The Task: "Upgrade" the system. <REDACTED> AI values Agentic Workflows, which means moving from a single prompt to a
   multi-step logic.

Possible Implementations:

Self-Correction: Have the LLM review its own output for errors.

Tool Use: Write a mock Python function (e.g., calculate_claim_total) and give the LLM instructions on when to
"call" it.

RAG / Context Injection: If they provide a small "knowledge base" file, you'll need to parse it and inject the
relevant snippet into the prompt.

Pro Tip: Be ready to use pydantic or structured JSON output. <REDACTED> engineers deal with structured data
(annotations/labels) constantly.

3. Final Evaluation & Debugging (Minutes 45-60)
   The Task: Run your improved agent on a larger set of data.

Debugging: The interviewer might point out a "regressive" error (e.g., "Your new prompt fixed the claim ID, but now it's failing to categorize the claim type").

Discussion: Talk about "Cost vs. Accuracy" tradeoffs—<REDACTED>'s enterprise customers care about token spend.

High-Value Technical Prep

| Skill         | What to focus on                                                                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Python SDKs   | Be very comfortable with `openai`, `anthropic`, or a generic wrapper like `langchain`/`instructor`. Know how to handle `rate_limit` or `json_mode` errors.                                                               |
| Pydantic      | Many <REDACTED> workflows rely on Structured Output. Practice defining a schema and having an LLM fill it.                                                                                                               |
| Data Cleaning | You might get messy JSON. Be fast with `json.loads()`, dict comprehensions, and handling `None` types.                                                                                                                   |
| LLM Debugging | If the model fails, don't just "try a different prompt." Explain your reasoning: "The model is suffering from 'lost in the middle' context issues," or "The system prompt is too ambiguous for this specific edge case." |
