"""
LLM integration for generating search queries using multiple backends

Supports multiple LLM backends:
- LM Studio: Local LLM server
- OpenWebUI: Open WebUI API
- Google AI Studio: Google Gemini API (cloud-based)

Configuration via environment variables:
- LLM_BACKEND: 'lm_studio', 'openwebui', or 'google_ai_studio' (default: 'lm_studio')
- LM_STUDIO_URL: LM Studio endpoint (default: http://localhost:1234/v1/chat/completions)
- LM_STUDIO_TIMEOUT: LM Studio timeout in seconds (default: 60)
- OPENWEBUI_URL: OpenWebUI endpoint (default: http://localhost:3000/api/chat/completions)
- OPENWEBUI_API_KEY: API key for OpenWebUI (required for authentication)
- OPENWEBUI_MODEL: Model name in OpenWebUI (default: 'llama3.2:3b')
- OPENWEBUI_TIMEOUT: OpenWebUI timeout in seconds (default: 120, increase for slower models)
- GOOGLE_AI_STUDIO_API_KEY: API key for Google AI Studio (get from https://aistudio.google.com/apikey)
- GOOGLE_AI_STUDIO_MODEL: Model name (default: 'gemini-1.5-flash', options: 'gemini-1.5-pro', 'gemini-2.0-flash-exp')
- GOOGLE_AI_STUDIO_TIMEOUT: Timeout in seconds (default: 60)

Common OpenWebUI endpoints to try:
- http://localhost:3000/api/chat/completions (OpenAI-compatible)
- http://localhost:3000/api/chat
- http://localhost:3000/ollama/v1/chat/completions (Ollama-compatible)
"""
import os
import requests
from typing import Optional

# Try to import Google Generative AI SDK (optional dependency)
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("WARNING: google-generativeai package not installed. Install with: pip install google-generativeai")
    print("Google AI Studio backend will not be available.")

# LLM Backend Configuration
# Options: 'lm_studio', 'openwebui', or 'google_ai_studio'
LLM_BACKEND = os.getenv('LLM_BACKEND', 'google_ai_studio').lower()

# LM Studio configuration
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', "http://localhost:1234/v1/chat/completions")
LM_STUDIO_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '60'))  # seconds

# OpenWebUI configuration
OPENWEBUI_URL = os.getenv('OPENWEBUI_URL', "http://localhost:8080/api/chat/completions")
OPENWEBUI_API_KEY = os.getenv('OPENWEBUI_API_KEY', '')
OPENWEBUI_MODEL = os.getenv('OPENWEBUI_MODEL', 'gemma3:12b')  # Model name in OpenWebUI
OPENWEBUI_TIMEOUT = int(os.getenv('OPENWEBUI_TIMEOUT', '180'))  # seconds (default 180 for reasoning models like DeepSeek-R1)

# Google AI Studio configuration
GOOGLE_AI_STUDIO_API_KEY = os.getenv('GOOGLE_AI_STUDIO_API_KEY', '')
GOOGLE_AI_STUDIO_MODEL = os.getenv('GOOGLE_AI_STUDIO_MODEL', 'gemini-2.5-flash')  # Older model, less strict safety filters
GOOGLE_AI_STUDIO_TIMEOUT = int(os.getenv('GOOGLE_AI_STUDIO_TIMEOUT', '60'))  # seconds

# Configure Google AI if available and API key is set
if GOOGLE_AI_AVAILABLE and GOOGLE_AI_STUDIO_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)
    except Exception as e:
        print(f"WARNING: Failed to configure Google AI Studio: {e}")

# Hardcoded instructions for each database (not editable by users)
DATABASE_INSTRUCTIONS = {
    'Pubmed': """Generate a PubMed Boolean search query for the given topic.

RULES:
- ONLY use concepts from the user's topic - do NOT add new concepts
- Identify 2-4 main concepts
- For each concept: include 2-6 keyword variants (synonyms, abbreviations)

FIELD TAGS (use both appropriately):
- [tiab] - for specific, important keywords in title/abstract (e.g., maternal[tiab], pregnancy[tiab], "oxidative stress"[tiab])
- [tw] - for general words, variants, synonyms, broader search (e.g., mother*[tw], gestation*[tw], oxidant*[tw])

GUIDELINES:
- Use [tiab] for main keywords and specific multi-word phrases
- Use [tw] for synonyms, variants, and truncated terms
- Use * for truncation (min 4 letters): keyword*[tw]
- Multi-word phrases: use quotes "phrase"[tiab]
- Group synonyms with OR in parentheses
- Combine concepts with AND
- Mix [tiab] and [tw] within each concept group

EXAMPLE:
("oxidative stress"[tiab] OR oxidant*[tw] OR "free radical"[tiab]) AND (maternal[tiab] OR mother*[tw] OR pregnan*[tw])

OUTPUT: Return ONLY the query, no explanations.""",

    'WOS': """Convert this search query to Web of Science format.

RULES:
- Remove all field tags like [tiab], [tw], [MeSH Terms]
- Keep all Boolean operators (AND, OR)
- Keep all parentheses grouping
- Keep all wildcards (*)
- Keep all quotes for phrases
- Can optionally use NEAR/x for proximity if it improves the query

OUTPUT: Return ONLY the adapted query, no explanations.""",

    'Scopus': """Convert this search query to Scopus format.

RULES:
- Remove all field tags like [tiab], [tw], [MeSH Terms]
- Keep all Boolean operators (AND, OR)
- Keep all parentheses grouping
- Keep all wildcards (*)
- Keep all quotes for phrases
- Can optionally use W/n or PRE/n for proximity if it improves the query
- Do NOT wrap in TITLE-ABS-KEY() - just convert the query syntax

OUTPUT: Return ONLY the adapted query, no explanations."""
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


        import time
        start_time = time.time()

        response = requests.post(
            OPENWEBUI_URL,
            json=payload,
            headers=headers,
            timeout=OPENWEBUI_TIMEOUT
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
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


def call_google_ai_studio(prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """
    Call Google AI Studio (Gemini) API to generate text

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (0.0-1.0, lower = more focused)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text from LLM
    """
    if not GOOGLE_AI_AVAILABLE:
        print("ERROR: Google Generative AI SDK not installed.")
        print("Install with: pip install google-generativeai")
        return None

    if not GOOGLE_AI_STUDIO_API_KEY:
        print("ERROR: GOOGLE_AI_STUDIO_API_KEY not set.")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Set it with: export GOOGLE_AI_STUDIO_API_KEY='your_key_here'")
        return None

    try:

        import time
        start_time = time.time()

        # Create the model
        model = genai.GenerativeModel(GOOGLE_AI_STUDIO_MODEL)

        # Configure generation settings
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Configure safety settings - BLOCK_NONE to allow all content
        # Gemini models only support 4 categories (PaLM categories cause errors)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        # Generate content with BLOCK_NONE safety settings
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            #safety_settings=safety_settings,
            request_options={'timeout': GOOGLE_AI_STUDIO_TIMEOUT}
        )

        elapsed_time = time.time() - start_time

        # Check if response was blocked by safety filters
        if response.prompt_feedback.block_reason:
            print(f"Google AI Studio: Prompt blocked by safety filters")
            print(f"Block reason: {response.prompt_feedback.block_reason}")
            print(f"Safety ratings: {response.prompt_feedback.safety_ratings}")
            return None

        # Extract text from response
        if response and response.text:
            generated_text = response.text.strip()
            return generated_text

        # Check if response has parts but finish_reason indicates issue
        if response.candidates:
            candidate = response.candidates[0]
            print(f"Google AI Studio: Response blocked or incomplete")
            print(f"Finish reason: {candidate.finish_reason}")
            if candidate.finish_reason == 2:  # SAFETY
                print(f"Safety ratings: {candidate.safety_ratings}")
            elif candidate.finish_reason == 3:  # RECITATION
                print(f"Content blocked due to recitation concerns")
            return None

        print(f"Google AI Studio API error: Empty response")
        if hasattr(response, 'prompt_feedback'):
            print(f"Prompt feedback: {response.prompt_feedback}")
        return None

    except Exception as e:
        error_message = str(e)
        print(f"Error calling Google AI Studio: {error_message}")

        # Provide helpful error messages
        if "API_KEY_INVALID" in error_message or "invalid api key" in error_message.lower():
            print("Your API key is invalid. Get a new one from: https://aistudio.google.com/apikey")
        elif "quota" in error_message.lower():
            print("API quota exceeded. Check your usage at: https://aistudio.google.com/")
        elif "timeout" in error_message.lower():
            print(f"Request timed out after {GOOGLE_AI_STUDIO_TIMEOUT}s")
            print("Try increasing timeout: export GOOGLE_AI_STUDIO_TIMEOUT=120")

        return None


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 200) -> str:
    """
    Call configured LLM backend (LM Studio, OpenWebUI, or Google AI Studio)

    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation (0.0-1.0, lower = more focused)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text from LLM
    """
    if LLM_BACKEND == 'openwebui':
        return call_openwebui(prompt, temperature, max_tokens)
    elif LLM_BACKEND == 'google_ai_studio':
        return call_google_ai_studio(prompt, temperature, max_tokens)
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
    generated_query = call_llm(prompt, temperature=0.3, max_tokens=5000)

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


def remove_field_tags(query: str) -> str:
    """
    Remove PubMed field tags from query using regex (faster and more reliable than LLM)

    Args:
        query: PubMed query with field tags like [tiab], [tw], [MeSH Terms]

    Returns:
        Query with field tags removed
    """
    import re

    # Remove field tags: [tiab], [tw], [MeSH Terms], [MeSH], etc.
    # Pattern matches: [any text inside square brackets]
    cleaned = re.sub(r'\[(?:tiab|tw|MeSH Terms|MeSH|all|au|ti|ab)\]', '', query)

    return cleaned.strip()


def generate_all_queries(name: str, context: dict = None) -> dict:
    """
    Generate search queries for all 3 databases at once.
    First generates PubMed, then adapts it for WOS and Scopus using regex (not LLM).

    Args:
        name: The research topic or keyword name
        context: Additional context (section, subsection, etc.)

    Returns:
        Dict with keys: Pubmed_Formula, WOS_Formula, Scopus_Formula,
                       Pubmed_Raw, WOS_Raw, Scopus_Raw
    """
    # Step 1: Generate PubMed query first
    pubmed_result = generate_search_query(name, 'Pubmed', context)


    # Step 2: If PubMed succeeded, use it as base for WOS and Scopus
    if pubmed_result['query'] and not pubmed_result['error_message']:
        # Adapt PubMed query for WOS and Scopus by removing field tags with regex
        # This ensures WOS and Scopus are IDENTICAL (just remove tags, keep everything else)
        wos_query = remove_field_tags(pubmed_result['query'])
        scopus_query = remove_field_tags(pubmed_result['query'])

        wos_result = {
            'query': wos_query,
            'raw': wos_query,
            'error_message': None
        }

        scopus_result = {
            'query': scopus_query,
            'raw': scopus_query,
            'error_message': None
        }
    else:
        # If PubMed failed, generate WOS and Scopus from scratch
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
        operator: Boolean operator (AND, OR)

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
