# Human-in-the-Loop (HITL) Marketplace Problem

Storing data isn't the challenge, it's actually about orchestrating human intelligence at <REDACTED> to feed the RLHF / GRPO loops.

## Requirements Gathering

### Functional requirements:

- Lab portal: AI labs must be able to upload "batches" of prompts and define "expert profiles" (i.e. PHD in Biology).
- Expert marketplace: A way for experts to see available tasks, claim them, and submit high-quality completions (responses, rankings, or reasoning traces).
- Matching Engine: Automatically route specific prompts, like Biology, to the correct pool of experts.
- Quality Control (QC): A workflow where a second expert or an LLM "judge" reviews the work before it's marked as "Gold Data".

### Non-Functional Requirements

- Scalabiity: Support millions of tasks and tens of thousands of concurrent experts.
- Durability: Data is the product. We can't lose a single expert completion.
- Auditability: Every change to a piece of training data must be versioned (lineage).
- Latency: The system shouldn't be the bottleneck, but "real-time" isn't the goal--throughput is.

## API Design

We need two sets of APIs: one ofr the Labs (Batch submission) and one for experts (task execution)

| Endpoint                | Method | Description                                                  |
| ----------------------- | ------ | ------------------------------------------------------------ |
| `/v1/batches`           | `POST` | Lab uploads a dataset requirement                            |
| `/v1/tasks/available`   | `GET`  | Expert fetches tasks filtered by their skill tags            |
| `/v1/tasks/{id}/claim`  | `POST` | "Locks" a task for a specific expert (prevents double-work). |
| `/v1/tasks/{id}/submit` | `POST` | Expert submits the reasoning / completion.                   |

## Data Schema

We need a relational structure to handle the complex state machine of a "Task":

- `Users`: `id, email, role (expert/lab admin), skills (i.e. ["biology", "python"])`.
- `Batches`: `id, lab_id, reward_criteria, status (pending/processing/completed)`.
- `Tasks`: `id, batch_id, prompt_text, status (available/claimed/completed/qc_failed)`.
- `Submissions`: `id, task_id, expert_id, completion_text, reasoning_trace, time_spent1`.

## High-Level Architecture

The "brain of the system" (task orchestrator):

1. Ingestion: labs drop large JSON/CSV files into S3. A metadata record is created in PostgreSQL.
2. Matching Engine: A service that maps `task.required_skills` to `expert_skills`. We can use an `Elasticsearch/Opensearch` index to query available tasks for experts quickly.
3. State Management: Use a Redis-based locking mechanism to ensure that when a domain expert clicks "Start Task", no one else can see it for a 30-minute window.
4. The Feedback Loop: Once submitted, the task moves to a `QC_PENDING` state. A separate worker assigns it to a "Reviewer".

## Deep Dive

### Data Quality at <REDACTED>

- Consistency: how do you handle two experts giving different answers?
  - Solution: consensus scoring. Assign the same task to 3 experts. If they disagree, escalate to a "Super-Expert".
- Observability: we need to monitor "Mean Time to Complete" (MTTC) and "Rejection Rate" per expert to prune low-quality contributors.
- Storage choice: PostgreSQL for the transactional state (Task Status, User profiles):
  - S3 for the heavy "blobs" (long LLM responses)
  - Vector DB (bonus) to find "duplicate prompts" or ensure we have a diverse distribution of questions.

### Matching Engine: Beyond Simple Search

Goal is to maximize expert utilization while ensuring task quality. A naive search is too slow and doesn't handle the nuances of expert skill levels.

The triage service: instead of expers just browsing, a modern data enigine uses a push/pull hybrid.

- The Pull (marketplace): experts browse an OpenSearch/Elasticsearch index to find tasks they feel confident in.
- Push (direct assignment): for high-priority batches (i.e. an AI lab needs 500 domain-specific responses in 2 hours), the system calculates a suitability score and pushes notifications directly to top-tier experts.

Suitability score can be thought of as:
`score = (expert_skill * task_difficulty) + recency_weight + historical_accuracy`

- Historical accuracy: a "reliability score" stored in the user profile, updated every time their work passes or fails QC.
- Skill Decay: if an expert hasn't done a task in 6 months, their weight drops slightly.

### The QC Workflow: The "Gold Standard" Loop

In generative AI, correctness is often subjective. To solve this, you need a multi-staged state machine.

#### Stage 1: the "trap" (pre-verification)

Before an expert starts a real task, the system injects a "Gold Task" -- a question where it already knows the answer. If the expert fails the "Gold Task", their current session is invalidated. This catches bots or experts who are speeding through.

#### Stage 2: Parallel Consensus (N-Way Review)

For critical data, you can't just trust one person.

- Task replication: assign the same prompt to 3 experts.
- Agreement logic: if all 3 provide similar answers/rankings, the task is marked as `AUTO_APPROVED`.
- Disagreement handling: if there's a "tie-break" needed (i.e. Expert A says the logic is sound, but expert B says it's flawed), the system automatically creates a QC task for a "senior reviewer"

#### Stage 3: The "Judge-in-the-Loop" (LLM-Assisted QC)

To <REDACTED>, we use AI to check the humans who are training the AI.

- An LLM (like a forzen version of Claude/GPT-4) acts as a linter. it checks:
  - Formatting errors (i.e. missing `<thought>` tags).
  - Tone consistency.
  - Plagiarism / AI-generated "cheating".

### System Architecture for QC & Matching

To keep the system response, these workflows are asynchronous:

1. Event Bus (Kafka / RabbitMQ): When an expert clicks "submit", a `TaskSubmitted` event is fired.
2. QC Orchestrator: A microservice listens to the event bus. It checks the "Task Policy" (i.e. "This batch requires 2-way consensus").
3. Dynamic Routing: If consensus isn't met, the Orchestrator calls the Matching Engine to find a new expert for a "Tie breaker" task.
4. Final sink: Once a task passes all gates, it moves to a Data Warehouse (BigQuery / Snowflake) where the AI lab can export for fine-tuning.

Tradeoffs:

- Consistency vs. Throughput: Do we wait for QC to finish before paying the expert (usually, we pay on submission but "claw back" / ban for poor QC performance to keep throughput high).
- State bloat: A single task can have 10+ versions of an answer. Use PostgreSQL JSONB (Json Binary) for the `submissions` table to store the evolution of the answer wihtout needing a 50 column table.

## Failure Modes

### Redis-lock expiry

Problem: An expert is deep in a 45-minute domain reasoning task. The Redis lock was set for 30 minutes, the lock expires, and the task is matched to another expert.

Solution: Use heartbeats from frontend/client to the server while the expert is active. The server extends the Redis TTL (time-to-live) automatically. If heartbeat stops, then the lock expires naturally.

### Poison Pill Prompt

Problem: A specific prompt is so broken / complex that every expert who tries it eventually "abandons" it / gets timed out.

Solution: Implement retry limits / dead letter queue (DLQ) for tasks. If a task is abandoned 3 times, move to a `MANUAL_REVIEW` status. An internal operations person investigates if the prompt is poorly written.

### Cheating Bot (Sybil Attack)

Problem: A user scripts a bot to submit AI-generated responses to farm rewards.

Solution: Statistical outlier detection. Monitor the `time_to_complete`. If an expert consistently submits 1,000 word-reasoning traces in 4 seconds, flag them for an immediate audit.

## Monitoring and Observability

### Infrastrucutre Level (health)

- Database bloat: since we use JSONB, we must monitor postgrew health. JSONB updates create "dead tuples" (old versions of the row). If cleanup can't keep up, disk usage spikes and queries slow down.
- Redis Memory: monitor eviction policy. If Redis runs out of RAM and starts evicting active task locks, you get race conditions.

### Service Level (Performance)

- Matching Latency: How long it takes for the Matching Engine to return results from OpenSearch.
- Event bus lag: Monitor consumer lag in Kafka/RabbitMQ for QC workers. If lag grows, experts don't get feedback on their work, leading to frustration.

### Product / Business Level

- Consensus rate: what % of tasks require a tie-breaker? If it's >20%, your expert pool may be low quality or instructions unclear.
- Expert throughput: Number of highly verified data points produced per hour.
- Reward Accuracy: Correlation between an expert's "Self-Reported Time" and the system's "Actual Active Time".
