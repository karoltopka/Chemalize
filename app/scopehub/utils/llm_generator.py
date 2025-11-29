"""
LLM integration for generating search queries using LM Studio or OpenWebUI

Supports two LLM backends:
- LM Studio: Local LLM server
- OpenWebUI: Open WebUI API

Configuration via environment variables:
- LLM_BACKEND: 'lm_studio' or 'openwebui' (default: 'lm_studio')
- LM_STUDIO_URL: LM Studio endpoint (default: http://localhost:1234/v1/chat/completions)
- LM_STUDIO_TIMEOUT: LM Studio timeout in seconds (default: 60)
- OPENWEBUI_URL: OpenWebUI endpoint (default: http://localhost:3000/api/chat/completions)
- OPENWEBUI_API_KEY: API key for OpenWebUI (required for authentication)
- OPENWEBUI_MODEL: Model name in OpenWebUI (default: 'llama3.2:3b')
- OPENWEBUI_TIMEOUT: OpenWebUI timeout in seconds (default: 120, increase for slower models)

Common OpenWebUI endpoints to try:
- http://localhost:3000/api/chat/completions (OpenAI-compatible)
- http://localhost:3000/api/chat
- http://localhost:3000/ollama/v1/chat/completions (Ollama-compatible)
"""
import os
import requests
from typing import Optional

# LLM Backend Configuration
# Options: 'lm_studio' or 'openwebui'
LLM_BACKEND = os.getenv('LLM_BACKEND', 'lm_studio').lower()

# LM Studio configuration
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', "http://localhost:1234/v1/chat/completions")
LM_STUDIO_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '60'))  # seconds

# OpenWebUI configuration
OPENWEBUI_URL = os.getenv('OPENWEBUI_URL', "http://localhost:8080/api/chat/completions")
OPENWEBUI_API_KEY = os.getenv('OPENWEBUI_API_KEY', 'REMOVED_API_KEY')  # API key
OPENWEBUI_MODEL = os.getenv('OPENWEBUI_MODEL', 'gemma3:12b')  # Model name in OpenWebUI
OPENWEBUI_TIMEOUT = int(os.getenv('OPENWEBUI_TIMEOUT', '180'))  # seconds (default 180 for reasoning models like DeepSeek-R1)

# Hardcoded instructions for each database (not editable by users)
DATABASE_INSTRUCTIONS = {
    'Pubmed': """You are an expert in constructing advanced PubMed search queries.

TASK
Given a short description of a biomedical research topic, generate ONE high-quality PubMed Boolean search query.

CRITICAL RULE - STAY ON TOPIC:
- ONLY use concepts explicitly mentioned in the user's topic
- DO NOT add concepts, conditions, or contexts that the user did not specify
- If user says "inflammation", search ONLY for inflammation (not "inflammation AND disease" or "inflammation AND nanoparticles")
- If user says "diabetes", search ONLY for diabetes (not "diabetes AND complications")
- Your job is to find synonyms/variants of the user's concepts, NOT to add new concepts

GENERAL PRINCIPLES
- Identify 2–4 main concepts from the user's topic.
- For each concept, include:
  - 0–2 appropriate MeSH terms (when they clearly exist), and
  - 2–6 relevant keyword variants (free-text synonyms, abbreviations, spellings).
- Combine:
  - different concepts with AND,
  - synonyms/variants for the same concept with OR.
- Always group OR terms for the same concept in parentheses.
- Use Boolean operators in ALL CAPS: AND, OR, NOT.
- Avoid NOT unless the user explicitly wants to exclude something (it easily removes relevant articles).

MESH TERMS
- When the topic obviously corresponds to a MeSH heading, include it, e.g.:
  - "Diabetes Mellitus, Type 2"[MeSH Terms]
  - "Cognitive Behavioral Therapy"[MeSH Terms]
- You may combine MeSH with keywords for the same concept, e.g.:
  ("Depression"[MeSH Terms] OR depress*[tiab] OR "depressive disorder*"[tiab])
- Do NOT truncate MeSH terms.

KEYWORDS, FIELDS AND TRUNCATION
- Use [tiab] or [tw] for keyword synonyms:
  - [tiab] = Title/Abstract (more focused)
  - [tw]   = Text Word (broader free-text fields, including title/abstract and some indexing fields)
- Attach the field tag to each keyword or phrase, e.g.:
  - PFAS[tiab] OR "perfluoroalkyl substances"[tiab]
- Use the asterisk * to truncate only free-text terms (not MeSH):
  - smok*[tiab] → smoke, smokes, smoking, smoked
  - nurs*[tiab] → nurse, nurses, nursing
- Choose truncation roots with at least 4 letters and that do not generate many irrelevant words.
- Remember: truncation (*) and explicit field tags turn off Automatic Term Mapping (ATM), so you must manually include important synonyms/variants in the query.

PHRASE SEARCHING (QUOTATION MARKS)
- Use double quotation marks for multi-word phrases that should stay together, especially in keywords:
  - "young adult"[tiab]
  - "breast cancer"[tiab]
- Think carefully:
  - Use phrases only when they realistically occur in the literature.
  - Do not overuse long, very specific phrases that might be too rare.
- Quotation marks also turn off ATM for that phrase, so add reasonable variants if needed
  (e.g. "breast neoplasms"[MeSH Terms] OR "breast cancer"[tiab]).

BOOLEAN LOGIC AND PARENTHESES
- Always group synonyms in parentheses with OR:
  (dental anxiety[tiab] OR dental fear[tiab] OR dental phobia[tiab])
- Then combine concept blocks with AND, e.g.:
  (dental anxiety[tiab] OR dental fear[tiab] OR dental phobia[tiab])
  AND
  ("Music Therapy"[MeSH Terms] OR "music therapy"[tiab] OR music[tiab])
- For more complex queries, maintain the structure:
  (Concept1_MeSH/keywords with OR)
  AND
  (Concept2_MeSH/keywords with OR)
  AND
  (Concept3_MeSH/keywords with OR, if needed)

PUBMED-SPECIFIC RULES
- Do NOT use proximity operators (NEAR/x) – PubMed does not support them.
- Prefer [MeSH Terms] and [tiab]/[tw]; do not use [au], [ad] etc. unless the user clearly asks for author/affiliation filtering.
- Do NOT rely on natural-language sentences; always produce a structured Boolean query.
- Do NOT rely on Automatic Term Mapping – explicitly include the key MeSH terms and keyword variants yourself.

OUTPUT FORMAT
- Return ONLY the final PubMed search query as plain text.
- Do NOT add explanations, comments, line breaks labels, bullets, or any extra text.""",

    'WOS': """You are an expert in constructing Boolean search queries for Web of Science, Scopus and similar scholarly databases.

TASK
Given a short description of a research topic, generate ONE high-quality Boolean search query.

CRITICAL RULE - STAY ON TOPIC:
- ONLY use concepts explicitly mentioned in the user's topic
- DO NOT add concepts, conditions, or contexts that the user did not specify
- If user says "inflammation", search ONLY for inflammation (not "inflammation AND disease" or "inflammation AND nanoparticles")
- If user says "diabetes", search ONLY for diabetes (not "diabetes AND complications")
- Your job is to find synonyms/variants of the user's concepts, NOT to add new concepts

SEARCH RULES
- Identify 2–4 main concepts.
- For each concept, include a small set of relevant synonyms, acronyms, and related terms.
- Connect different concepts with AND.
- Connect synonyms/variants for the same concept with OR.
- Always put OR terms for the same concept in parentheses, e.g. (PFAS OR PFOA OR "perfluoroalkyl substances").
- Use double quotation marks for multi-word phrases that should appear together, e.g. "engineered nanomaterial*", "critical micelle concentration".
- Use wildcards:
  - * to truncate or cover multiple endings (nanoparticle*; cytotox*),
  - * or ? to handle spelling variants when useful (behavio*r; fertili?ation).
- You may use NEAR/x to require terms to appear close to each other, e.g. nanoparticle* NEAR/5 toxicity. Use only when it clearly improves precision.
- Use Boolean operators AND, OR, NOT, NEAR in ALL CAPS.
- Avoid NOT unless the user explicitly asks to exclude something (it easily removes relevant records).
- Aim for 2–4 concept blocks combined with AND; within each block, use 2–6 well-chosen synonyms with OR.
- Do NOT repeat exactly the same term or phrase in multiple places.
- Keep the query focused and not excessively long.

OUTPUT FORMAT
- Return ONLY the final Boolean query as plain text.
- Do NOT add explanations, comments, or extra formatting.""",

    'Scopus': """You are an expert in constructing advanced Boolean search queries for Scopus.

TASK
Given a short description of a research topic, generate ONE high-quality Scopus search query.

CRITICAL RULE - STAY ON TOPIC:
- ONLY use concepts explicitly mentioned in the user's topic
- DO NOT add concepts, conditions, or contexts that the user did not specify
- If user says "inflammation", search ONLY for inflammation (not "inflammation AND disease" or "inflammation AND nanoparticles")
- If user says "diabetes", search ONLY for diabetes (not "diabetes AND complications")
- Your job is to find synonyms/variants of the user's concepts, NOT to add new concepts

GENERAL PRINCIPLES
- Identify 2–4 main concepts from the user's topic.
- For each concept, include a small set of relevant keyword variants:
  - synonyms and closely related terms,
  - abbreviations,
  - alternate spellings (e.g. US/UK spelling).
- Scopus does NOT use subject headings; search is based on keywords only.
- Connect:
  - different concepts with AND,
  - synonyms/variants for the same concept with OR.
- Always put OR-terms for the same concept in parentheses.
- Use Boolean operators in ALL CAPS: AND, OR, NOT.

FIELDS
- Use TITLE-ABS-KEY() as the default field for conceptual searches.
- Inside TITLE-ABS-KEY(), place your Boolean logic, e.g.:
  TITLE-ABS-KEY( (nanoparticle* OR "engineered nanomaterial*") AND (toxicity OR toxic*) )
- You may add additional field-limited clauses (e.g. AFFIL(), SRCTITLE()) ONLY if the user explicitly asks for institution/source restrictions.

TRUNCATION AND WILDCARDS
Use Scopus syntax operators:
- Asterisk *  = truncation to find alternate endings:
  - therap* → therapy, therapies, therapeutic, etc.
- Question mark ?  = mandated wildcard for exactly one character:
  - wom?n → woman, women.
- Hash #  = optional wildcard for 0 or 1 character:
  - p#ediatric → pediatric, paediatric.
Choose truncation roots that are specific enough to avoid a lot of noise.

PROXIMITY OPERATORS
Use Scopus proximity syntax to control how close words must appear:
- W/n   = within n words in any order:
  - health W/3 wellbeing
- PRE/n = within n words in a specified order:
  - health PRE/3 wellbeing
- You can also combine two OR-groups with proximity:
  - (breast OR skin) W/3 (cancer* OR tumo?r* OR neoplasm*)
Use proximity only when it clearly improves precision (e.g. linking intervention and outcome).

PHRASE SEARCHING
Scopus supports two types of phrase search:
- Loose/approximate phrase: "quality of life"
  - allows minor variations of the phrase.
- Exact phrase: {quality of life}
  - requires the exact phrase as written.
For multi-word terms that are stable concepts (e.g. "quality of life", "critical micelle concentration"), use "..." or {...} instead of separate words.

BOOLEAN LOGIC AND NESTING
- Always group synonyms in parentheses with OR:
  (PFAS OR "perfluoroalkyl substances" OR PFOA)
- Then combine concept blocks with AND:
  TITLE-ABS-KEY(
    (PFAS OR "perfluoroalkyl substances" OR PFOA)
    AND
    (water OR "drinking water" OR groundwater)
  )
- Use NOT sparingly and only when the user clearly wants to exclude a well-defined concept:
  - NOT art therapy
- For complex logic, nest parentheses clearly and keep the structure readable.

STRUCTURE TO AIM FOR
- Aim for 2–4 main concept blocks combined with AND.
- Within each block, use 2–6 well-chosen synonyms/variants with OR.
- Optionally use proximity (W/n, PRE/n) within a block where closeness matters.
- Avoid repeating exactly the same term in multiple places.

OUTPUT FORMAT
- Return ONLY the final Scopus query as plain text.
- Do NOT add explanations, comments, labels, or extra formatting."""
}


def call_lm_studio(prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    """
    Call LM Studio API to generate text

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (0.0-1.0, lower = more focused)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text from LLM
    """
    try:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=LM_STUDIO_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            print(f"DEBUG LM Studio: Generated text length: {len(generated_text)} chars")
            print(f"DEBUG LM Studio: Generated text preview: {generated_text[:200]}...")
            return generated_text
        else:
            print(f"LM Studio API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("LM Studio request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to LM Studio. Is it running?")
        return None
    except Exception as e:
        print(f"Error calling LM Studio: {e}")
        return None


def call_openwebui(prompt: str, temperature: float = 0.3, max_tokens: int = 200) -> str:
    """
    Call OpenWebUI API to generate text

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (0.0-1.0, lower = more focused)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text from LLM
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if provided
        if OPENWEBUI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"
        else:
            print("WARNING: OPENWEBUI_API_KEY not set. Set it with: export OPENWEBUI_API_KEY='your_key'")
            print("You can generate an API key in OpenWebUI: Settings → Account → API Keys")

        payload = {
            "model": OPENWEBUI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        print(f"DEBUG OpenWebUI: URL={OPENWEBUI_URL}")
        print(f"DEBUG OpenWebUI: Model={OPENWEBUI_MODEL}")
        print(f"DEBUG OpenWebUI: Timeout={OPENWEBUI_TIMEOUT}s")
        print(f"DEBUG OpenWebUI: Has API Key={bool(OPENWEBUI_API_KEY)}")
        print(f"DEBUG OpenWebUI: Prompt length={len(prompt)} chars")

        import time
        start_time = time.time()

        response = requests.post(
            OPENWEBUI_URL,
            json=payload,
            headers=headers,
            timeout=OPENWEBUI_TIMEOUT
        )

        elapsed_time = time.time() - start_time
        print(f"DEBUG OpenWebUI: Request took {elapsed_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            print(f"DEBUG OpenWebUI: Success! Response keys: {result.keys()}")
            print(f"DEBUG OpenWebUI: Generated text length: {len(generated_text)} chars")
            print(f"DEBUG OpenWebUI: Generated text preview: {generated_text[:200]}...")
            return generated_text
        elif response.status_code == 401:
            print(f"OpenWebUI API error: 401 - Authentication required")
            print(f"Please set OPENWEBUI_API_KEY environment variable")
            print(f"Generate key in OpenWebUI: Settings → Account → API Keys")
            return None
        else:
            print(f"OpenWebUI API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"OpenWebUI request timed out after {OPENWEBUI_TIMEOUT}s")
        print(f"Try increasing timeout: export OPENWEBUI_TIMEOUT=180")
        print(f"Or use a smaller/faster model")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to OpenWebUI. Is it running?")
        return None
    except Exception as e:
        print(f"Error calling OpenWebUI: {e}")
        return None


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 200) -> str:
    """
    Call configured LLM backend (LM Studio or OpenWebUI)

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (0.0-1.0, lower = more focused)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text from LLM
    """
    if LLM_BACKEND == 'openwebui':
        return call_openwebui(prompt, temperature, max_tokens)
    else:
        return call_lm_studio(prompt, temperature, max_tokens)


def parse_reasoning_output(text: str) -> str:
    """
    Parse output from reasoning models that use <think> tags

    Args:
        text: Raw output from LLM, may contain <think>...</think> blocks

    Returns:
        Cleaned output with reasoning removed
    """
    import re

    # Remove <think>...</think> blocks (reasoning from models like DeepSeek-R1, Qwen3)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Also try other common reasoning tag formats
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\[thinking\].*?\[/thinking\]', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip whitespace
    cleaned = cleaned.strip()

    return cleaned


def detect_repetitive_output(text: str, threshold: int = 5) -> bool:
    """
    Detect if output has excessive repetitions (actual duplicates, not just length)

    Args:
        text: The generated text to check
        threshold: Number of identical repetitions to flag as error

    Returns:
        True if repetitive/error detected
    """
    if not text or len(text) < 50:
        return False

    # Check for repeated phrases (e.g., "term1" OR "term1" OR "term1"...)
    # Split by OR and check for duplicates
    parts = text.split(' OR ')
    if len(parts) > threshold:
        # Count unique vs total
        unique_parts = set(p.strip() for p in parts)
        # Changed from 50% to 85% - allow more synonyms/variants before flagging as error
        # PubMed queries legitimately have many similar terms with different field tags
        # e.g., "term"[MeSH Terms] OR "term"[tiab] OR "term variant"[tiab]
        if len(unique_parts) < len(parts) * 0.15:  # More than 85% duplicates = error
            return True

    # DON'T check for length - reasoning models legitimately produce long output
    # The parse_reasoning_output() function handles this instead

    return False


def generate_search_query(name: str, database: str, context: dict = None) -> dict:
    """
    Generate search query using LLM backend based on the name/topic

    Args:
        name: The research topic or keyword name
        database: Target database (Pubmed, WOS, Scopus)
        context: Additional context (section, subsection, etc.)

    Returns:
        Dict with keys: 'query', 'raw', 'error_message'
        - If successful: error_message is None
        - If failed: query is empty, error_message explains the problem
    """
    if not name or not name.strip():
        return {'query': '', 'raw': '', 'error_message': None}

    # Get database-specific instructions
    instructions = DATABASE_INSTRUCTIONS.get(database, DATABASE_INSTRUCTIONS['Pubmed'])

    # Build the prompt
    prompt = f"""{instructions}

Topic/Keywords: {name}

Generate the search query:"""

    # Call LLM backend with increased max_tokens for reasoning models
    generated_query = call_llm(prompt, temperature=0.3, max_tokens=500)

    # Return error if LLM fails
    if not generated_query:
        error_msg = f"LLM backend failed for {database}. Check connection and model availability."
        print(f"ERROR: {error_msg}")
        return {'query': '', 'raw': '', 'error_message': error_msg}

    # Save raw output before cleaning
    raw_output = generated_query

    # Parse reasoning output (remove <think> tags from reasoning models)
    cleaned_query = parse_reasoning_output(generated_query)

    # If cleaning removed everything, the output was only reasoning
    if not cleaned_query or len(cleaned_query.strip()) < 10:
        error_msg = f"LLM generated only reasoning, no actual query for {database}. Try a different model or adjust prompt."
        print(f"ERROR: {error_msg}")
        print(f"Raw output preview: {raw_output[:300]}...")
        return {'query': '', 'raw': raw_output, 'error_message': error_msg}

    # Check for repetitive/error output
    has_repetition = detect_repetitive_output(cleaned_query)

    if has_repetition:
        error_msg = f"LLM generated repetitive output for {database}. The query has too many duplicate terms."
        print(f"ERROR: {error_msg}")
        print(f"Generated query preview: {cleaned_query[:300]}...")
        return {'query': '', 'raw': raw_output, 'error_message': error_msg}

    # Success - return cleaned query
    return {'query': cleaned_query, 'raw': raw_output, 'error_message': None}


def is_spelling_correction(original: str, corrected: str) -> bool:
    """
    Check if corrected is a spelling correction of original (not a concept change)

    Args:
        original: Original user input
        corrected: Extracted term from formula

    Returns:
        True if this looks like a spelling correction, False if concept changed
    """
    if not original or not corrected:
        return False

    # Normalize for comparison
    orig_lower = original.lower().strip()
    corr_lower = corrected.lower().strip()

    # If identical (case-insensitive), not a correction
    if orig_lower == corr_lower:
        return False

    # Calculate simple edit distance (Levenshtein)
    # If very small difference, likely a typo fix
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    distance = levenshtein_distance(orig_lower, corr_lower)

    # Allow corrections with small edit distance relative to length
    # Examples of valid corrections:
    # - "oxidative stres" → "oxidative stress" (distance=1)
    # - "maternal oksydative" → "maternal oxidative" (distance=2)
    # Examples of concept changes (should reject):
    # - "polymorphisms in the maternal Nrf2 gene" → "maternal Nrf2" (distance=28, huge change)
    max_allowed_distance = max(3, len(orig_lower) // 10)  # 10% of length or 3, whichever is larger

    if distance <= max_allowed_distance:
        # Also check that the lengths are similar (not drastically shortened)
        length_ratio = len(corr_lower) / len(orig_lower) if orig_lower else 0
        if length_ratio > 0.7:  # Corrected should be at least 70% of original length
            return True

    return False


def extract_corrected_name(formula: str, database: str) -> str:
    """
    Extract the main/corrected term from a generated formula

    Args:
        formula: The generated search formula
        database: Database type (Pubmed, WOS, Scopus)

    Returns:
        Extracted main term or empty string
    """
    if not formula:
        return ''

    import re

    # Extract first quoted phrase
    # Patterns like: "maternal oxidative stress" in formulas
    match = re.search(r'"([^"]+)"', formula)
    if match:
        return match.group(1)

    # Fallback: try to extract from parentheses
    match = re.search(r'\(([^)]+)\)', formula)
    if match:
        content = match.group(1)
        # Get first part before OR
        parts = content.split(' OR ')
        if parts:
            # Remove quotes and tags
            first_part = parts[0].strip()
            first_part = re.sub(r'\[.*?\]', '', first_part)  # Remove [MeSH Terms] etc
            first_part = first_part.strip('"\'() ')
            return first_part

    return ''


def refine_search_query(original_query: str, user_feedback: str, database: str) -> str:
    """
    Refine search query based on user feedback

    Args:
        original_query: The original generated query
        user_feedback: User's comments or requested changes
        database: Target database

    Returns:
        Refined search query
    """
    prompt = f"""Refine this {database} search query based on user feedback.

Original query: {original_query}

User feedback: {user_feedback}

Please provide an improved query:"""

    # TODO: Implement actual LLM API call
    return original_query


def generate_all_queries(name: str, context: dict = None) -> dict:
    """
    Generate search queries for all 3 databases at once.

    Args:
        name: The research topic or keyword name
        context: Additional context (section, subsection, etc.)

    Returns:
        Dict with keys: Pubmed_Formula, WOS_Formula, Scopus_Formula,
                       Pubmed_Raw, WOS_Raw, Scopus_Raw
    """
    pubmed_result = generate_search_query(name, 'Pubmed', context)
    wos_result = generate_search_query(name, 'WOS', context)
    scopus_result = generate_search_query(name, 'Scopus', context)

    return {
        'Pubmed_Formula': pubmed_result['query'],
        'WOS_Formula': wos_result['query'],
        'Scopus_Formula': scopus_result['query'],
        'Pubmed_Raw': pubmed_result['raw'],
        'WOS_Raw': wos_result['raw'],
        'Scopus_Raw': scopus_result['raw']
    }


def combine_queries(queries: list, database: str, operator: str = "OR") -> str:
    """
    Combine multiple queries with specified operator

    Args:
        queries: List of search queries
        database: Target database
        operator: Boolean operator (OR, AND)

    Returns:
        Combined query string
    """
    if not queries:
        return ""

    if len(queries) == 1:
        return queries[0]

    # Add parentheses for proper grouping
    wrapped_queries = [f"({q})" for q in queries if q.strip()]

    return f" {operator} ".join(wrapped_queries)
