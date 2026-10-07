def inject_reasoning_effort(messages, effort_level):
    """
    messages: list of dicts with keys 'role' and 'content'
    effort_level: one of 'low', 'medium', 'high', 'max'
    Returns: new list of message dicts with reasoning-effort system prompt injected.
    """

    effort_level_instructions = {
        "low": "Reasoning effort: low. Provide concise answers with minimal deliberation.",
        "medium": "Reasoning effort: medium. Think step by step before answering.",
        "high": "Reasoning effort: high. Carefully decompose the problem and verify each step.",
        "max": "Reasoning effort: max. Exhaustively decompose the problem, stress-test all edge cases, and document every intermediate step."
    }

    if effort_level not in effort_level_instructions:
        raise ValueError()

    messages = messages.copy()

    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = effort_level_instructions[effort_level] + "\n" + messages[0]['content']
    else:
        messages = [{"role": "system", "content": effort_level_instructions[effort_level]}] + messages
    
    return messages