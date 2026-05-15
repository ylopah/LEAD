import re
import logging
import json
from openai import OpenAI

logger = logging.getLogger("Extraction")


def send_query_to_openai(message, model, api_config):
    """General API request function for LLM communication."""
    try:
        client = OpenAI(
            api_key=api_config['key'],
            base_url=api_config['base_url'],
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": str(message)}],
            temperature=0.1,
            top_p=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API Error (Model: {model}): {e}")
        return "{}"


def pre_filter_doc_by_keywords(doc_content, variables, min_keyword_hits=1):
    """Pre-filter document paragraphs: only keep those mentioning target variables.
    Returns shortened content string focused on relevant passages."""
    if isinstance(doc_content, list):
        paragraphs = doc_content
    else:
        paragraphs = [str(doc_content)]

    var_terms = [v.lower() for v in variables.keys()]
    relevant_paragraphs = []
    for p in paragraphs:
        p_lower = p.lower()
        if any(v in p_lower for v in var_terms):
            relevant_paragraphs.append(p)
            if len(" ".join(relevant_paragraphs)) >= 6000:
                break

    if len(relevant_paragraphs) < min_keyword_hits:
        return ""  # Signal: doc doesn't mention any variable

    return " ".join(relevant_paragraphs)[:6000]


def extract_verification_result(response):
    """Extract binary veracity labels from LLM responses."""
    res = response.lower()
    if "the veracity of claim" in res:
        relevant_part = res.split("the veracity of claim")[-1]
        if "true" in relevant_part:
            return "True"
        if "false" in relevant_part:
            return "False"
    if "true" in res:
        return "True"
    if "false" in res:
        return "False"
    return "Unknown"


def extract_batch_verification_results(response, claim_a, claim_b):
    """Extract veracity for both directional claims from a batch response."""
    res = response.lower()
    results = {claim_a: "Unknown", claim_b: "Unknown"}

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for key in data:
                val = str(data[key]).lower()
                if "true" in val:
                    results[key] = "True"
                elif "false" in val:
                    results[key] = "False"
                else:
                    results[key] = "Unknown"
            return results
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: try to parse "A causes B is True/False" patterns
    for claim in [claim_a, claim_b]:
        pattern = re.escape(claim.lower())
        if re.search(pattern + r".*?\btrue\b", res):
            results[claim] = "True"
        elif re.search(pattern + r".*?\bfalse\b", res):
            results[claim] = "False"

    return results


def extract_value_of_variables(doc, variables, model, api_config):
    """Batch extract variable states (presence/absence) from a document."""
    doc_str = pre_filter_doc_by_keywords(doc, variables)
    if not doc_str:
        return {var: "False" for var in variables.keys()}

    var_list_str = ", ".join(variables.keys())

    query = f"""
Given the following document:
{doc_str}
Task: Determine the True/False status for ALL these variables: [{var_list_str}].
- True: The variable is mentioned as present, active, or a confirmed factor.
- False: The variable is explicitly absent, denied, or not mentioned at all.

Return the result ONLY as a strictly formatted JSON object.
Example: {{"var_a": "True", "var_b": "False"}}
"""
    response = send_query_to_openai(query, model, api_config)

    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        extracted_data = json.loads(json_str)
        final_results = {
            var: "True" if "true" in str(extracted_data.get(var, "False")).lower() else "False"
            for var in variables.keys()
        }
        return final_results
    except Exception:
        logger.warning("JSON parsing failed for a document, using default False.")
        return {var: "False" for var in variables.keys()}


def causal_claim_verification(claim, docs, model, api_config):
    """Verify a single causal claim against retrieved documents (original, kept for compatibility)."""
    all_veracity = []
    for url, doc_content in docs.items():
        logger.debug(f"Verifying {claim} via {url}")
        doc_str = " ".join(doc_content) if isinstance(doc_content, list) else str(doc_content)
        doc_str = doc_str[:6000]

        query = f"""
Document Content: {doc_str}

Scientific Claim to Verify: "{claim}"

Instructions:
1. Does the text provide evidence that one variable influences the other?
2. Look for keywords like "leads to", "associated with", "risk factor for", "precedes".
3. If there is strong evidence for the claim, answer 'True'.
4. If the text explicitly contradicts the claim or suggests the opposite direction, answer 'False'.
5. If the document mentions both variables but doesn't explain their relationship, or doesn't mention them at all, answer 'Unknown'.

Output Format: The veracity of claim '{claim}' is [True/False/Unknown].
Reason: (One short sentence)
"""
        response = send_query_to_openai(query, model, api_config)
        veracity = extract_verification_result(response)
        all_veracity.append(veracity)
    return all_veracity


def causal_claim_verification_batch(pair, docs, model, api_config):
    """Verify BOTH directional claims for a variable pair in a single API call.
    Reduces API calls by 50% compared to verifying each direction separately.

    Returns: dict with keys like "A causes B" -> ["True"/"False"/"Unknown", ...]
    """
    claim_a = f"{pair[0]} causes {pair[1]}"
    claim_b = f"{pair[1]} causes {pair[0]}"

    all_results = {claim_a: [], claim_b: []}

    for url, doc_content in docs.items():
        logger.debug(f"Batch verifying {pair} via {url}")
        doc_str = " ".join(doc_content) if isinstance(doc_content, list) else str(doc_content)
        doc_str = doc_str[:6000]

        query = f"""
Document Content: {doc_str}

Two Scientific Claims to Verify:
1. "{claim_a}"
2. "{claim_b}"

For each claim, determine if the document provides evidence:
- 'True': strong evidence supporting the claim
- 'False': evidence contradicting the claim or suggesting the opposite
- 'Unknown': variables mentioned but relationship unclear, or variables not mentioned

Look for keywords like "leads to", "associated with", "risk factor for", "causes", "precedes".

Return ONLY a JSON object:
{{"{claim_a}": "True/False/Unknown", "{claim_b}": "True/False/Unknown"}}
"""
        response = send_query_to_openai(query, model, api_config)
        batch_results = extract_batch_verification_results(response, claim_a, claim_b)
        all_results[claim_a].append(batch_results[claim_a])
        all_results[claim_b].append(batch_results[claim_b])

    return all_results


def get_authoritative_domains(dataset_name, nodes, model_func, model_name):
    """Retrieve highly credible web domains for specialized searches."""
    prompt = f"""
You are a research assistant. We are conducting a causal discovery study on the '{dataset_name}' dataset.
The variables involved are: {', '.join(nodes)}.
Please list 5-8 highly authoritative, official, or academic domains (e.g., cdc.gov, who.int, nih.gov)
that are most likely to contain rigorous clinical data or peer-reviewed research about these variables.
Return ONLY the domain names separated by commas.
"""
    response = model_func(prompt, model_name)
    domains = [d.strip() for d in response.split(",") if "." in d]
    logger.info(f"Authoritative domains identified: {domains}")
    return domains
