# System Design

## Recruiter Notes

- standard backend product focused system design
- Go on Google / YouTube
- Based on <REDACTED> AI products. This has to do with their generative ai data engine
- Gen ai data engine:
- Fine tuning product
- Two sided marketplace with AI labs and contributors / experts
- Lab submits prompt and use cases to the data engine
- They allocate the appropriate expert to particular use cases
- An AI lab wants their LLM to be better at a specific domain (e.g., coding, law) so they source experts to help with this task and make changes / recommendations
- They take the high quality training data that can be applied to the next gen model
- Generating high quality training data to be applied to LLMs.
- Come prepared with whiteboarding tools (I.e. Excalidraw).
- API design + schema design + general architecture
- Start with functional and nonfunctional requirements
- Want to see product requirements and system requirements
- Scalability, latency, durability, fault tolerance
- Database: provides what db needs to store, db choice, and why.
- Observability / monitoring — infra, service, app level (bonus topics)
- Really focus product and system requirements (80% breadth, 20% depth)

---

## Interview Platform Notes

You are expected to architect a system at a high level. Rather than writing out code line by line, you'll be asked to consider broad design decisions:

Understand System Requirements: Familiarize yourself with general performance needs, real-time operations, and periodic update mechanisms.
Master High-Level Architecture: Be able to outline core components like processing services, detection services, data storage, and model updating processes.
Develop Design Strategies: Practice designing systems using advanced processing models, detection techniques, and regular model updates.
Plan API Endpoints: Prepare to discuss different API endpoints for real-time, batch processing, and status checks, focusing on their purpose and implementation.
Consider Key Factors: Be ready to address performance, scalability, error handling, and security in your system design, explaining your approach to each.

A few tips for this interview:

Talk through your thought process. In our interviews, our engineers are evaluating not only your technical abilities, but also how you approach and solve problems.
Ask clarifying questions if you do not understand the problem or need more information. Our System Design interviews are deliberately open-ended, because our engineers are looking to see how you engage the problem. In particular, they are looking to see which areas you identify as the most important piece of the technological puzzle you've been presented.
Think about ways to improve the solution you'll present. In many cases, the first answer that comes to mind may not be the most elegant solution, and could need some refining. It's definitely worthwhile to talk through your initial thoughts to a question, and take time to compose a more efficient solution.

You should come prepared to work with a whiteboarding tool of your choice.

---

Notes from ML fundamentals interviewer:

- Similar questions to previous interviewer -> scenario deploying model for client + understand technicality
  - 3 nodes with different servers + how to balance / implement it:
    - Unsure on optimal solution.
    - Higher-level ideas: geogrpahical proximity to servers + send to remaining ones if overloaded. You may also need a node balancer after the geographical ones.
  - Locality based node balancing.
  - Something breaks.
  - Fast inference.
