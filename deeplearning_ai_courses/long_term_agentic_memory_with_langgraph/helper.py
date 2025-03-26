# Format list of few shots
def format_few_shot_examples(examples):
    # Template for formating an example to put in prompt
    template = """Email Subject: {subject}
        Email From: {from_email}
        Email To: {to_email}
        Email Content: 
        ```
        {content}
        ```
        > Triage Result: {result}"""

    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


def get_create_prompt_func(profile, system_prompt, prompt_instructions):
    return lambda state: [
        {
            "role": "system", 
            "content": system_prompt.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']