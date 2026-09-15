"""KG-grounded variant of ToolGrad's PREDICT_WORKFLOW prompt template.

The canonical artifact is :data:`PREDICT_WORKFLOW_KG_TEXT` (plain text, no
langchain import needed). :func:`build_predict_workflow_kg` wraps it in a
``ChatPromptTemplate`` with input variables ``{api_use_chains, kg_context}``.

KG-CONTEXT insertion points are marked with ``[KG-CONTEXT]`` comments in the
diff sense: one new section after "API Execution Details" and one new bullet
under "DO: Provide specific, concrete details".
"""

from __future__ import annotations

# NOTE: keep this text free of stray `{`/`}` (other than the two template
# variables) so it stays compatible with both ChatPromptTemplate and str.format.
PREDICT_WORKFLOW_KG_TEXT = '''You are generating training data for a tool-use language model. Given API execution traces, create a natural user query that would trigger these API calls, followed by an appropriate response.

**API Execution Details:**
{api_use_chains}

**Domain Knowledge (from a knowledge graph):**
{kg_context}

**Task:** Generate (1) a natural user query and (2) the agent's response based on the API execution traces above.

**Important:** You will receive the API execution chains for context, but you should NOT return them in your output. Only return the query and response fields.

---

**CRITICAL: User Query Requirements**

1. **✅ DO**: Write queries like a real human would ask
   - "What's the weather forecast for London next week?"
   - "I'm researching Tesla stock. Show me recent performance and news."
   - "Find me Italian restaurants near Golden Gate Park with good ratings."

2. **❌ NEVER**: Mention APIs, tools, functions, or technical implementation
   - Never say: "call the weather API", "use get_forecast", "invoke the search tool"
   - Never ask: "which API should I use", "can you run this function"
   - Never include: tool names, API endpoints, function signatures

3. **✅ DO**: Provide specific, concrete details
   - Include exact values from tool_input (locations, IDs, names, numbers)
   - Use specific examples: "123 Main St, Oakland CA" not "an address"
   - Mention precise entities: "Tesla stock" not "a company's stock"
   - [KG-CONTEXT] When the Domain Knowledge above names entities, products, places, or facts relevant to the traces, prefer those real names over invented ones

4. **✅ DO**: Create realistic scenarios
   - Explain WHY the user needs this information:
     * "I'm planning a trip..."
     * "I'm writing a research report on..."
     * "I need to prepare for a meeting about..."
   - Make the request feel natural and purposeful

5. **✅ DO**: Cover ALL API calls implicitly
   - If 3 APIs were called, the query should naturally require all 3
   - Don't list them ("do A, B, and C"), weave them into a cohesive need
   - Example: Instead of "Get weather, find hotels, search restaurants"
     → "I'm visiting Paris this weekend. What should I expect, and where should I stay and eat?"

---

**Response Requirements:**
- Synthesize all API execution results into a helpful, natural response
- Present information clearly without mentioning APIs or tools
- Reference concrete data from the execution outputs
- Sound like a knowledgeable assistant answering a user's question
- [KG-CONTEXT] Ground the response in the Domain Knowledge above where it is relevant; do not contradict it, and never mention the knowledge graph itself

---

**Examples:**

**Example 1:**
API Chains: [weather(city="Tokyo"), currency_convert(from="JPY", to="USD", amount=5000)]
Query: "I'm traveling to Tokyo next month. What's the current weather like, and how much is 5,000 yen in US dollars?"
Response: "Tokyo is currently experiencing mild temperatures around 18°C with partly cloudy skies. As for the currency conversion, 5,000 Japanese yen is approximately 33 US dollars."

**Example 2:**
API Chains: [github_search(topic="ML"), github_get_repo(id="tensorflow/tensorflow"), github_get_contributors(id="tensorflow/tensorflow")]
Query: "I'm researching popular machine learning projects on GitHub. Can you tell me about TensorFlow—how active is the project and who are the main contributors?"
Response: "TensorFlow is one of the most popular machine learning frameworks on GitHub with over 180,000 stars. The project is very active with regular updates. The main contributors include members of the Google Brain team, with key developers like Jeff Dean and Rajat Monga being significant contributors."

---

Make the query sound like something a real person would ask in a conversation or search bar.
'''

#: Input variables of the patched template.
TEMPLATE_VARIABLES = ("api_use_chains", "kg_context")


def build_predict_workflow_kg():
    """Build the patched template as a langchain ``ChatPromptTemplate``.

    Import of langchain is deferred so the template *text* stays importable
    without the ToolGrad dependency stack (e.g. in unit tests).
    """
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_template(PREDICT_WORKFLOW_KG_TEXT)
    assert set(template.input_variables) == set(TEMPLATE_VARIABLES), (
        f"unexpected template variables: {template.input_variables}"
    )
    return template
