"""Small helpers for displaying numbered person entity tokens as stable names."""

from __future__ import annotations

import re
from typing import Any


PERSON_TOKEN_RE = re.compile(r"\[PERSON_NAME_(\d+)\]")
PERSON_DISPLAY_NAMES = ("Alex", "Robin", "Casey")


def registered_person_tokens(tokenizer: Any) -> dict[str, int]:
    return {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in (getattr(tokenizer, "all_special_tokens", None) or [])
        if PERSON_TOKEN_RE.fullmatch(token)
    }


def render_person_tokens(text: str) -> tuple[str, dict[str, str]]:
    names = {
        f"[PERSON_NAME_{number}]": name
        for number, name in enumerate(PERSON_DISPLAY_NAMES, start=1)
        if f"[PERSON_NAME_{number}]" in text
    }
    rendered = PERSON_TOKEN_RE.sub(lambda match: names.get(match.group(0), match.group(0)), text)
    return rendered, names


def decode_story(token_ids: list[int], tokenizer: Any, eos_ids: list[int]) -> tuple[str, str, dict[str, str] | None]:
    person_tokens = registered_person_tokens(tokenizer)
    present_tokens = {token for token, token_id in person_tokens.items() if token_id in token_ids}
    type_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in tokenizer.all_special_tokens
        if re.fullmatch(r"\[STORY_TYPE_[1-6]\]", token)
    }
    if not present_tokens and not any(token_id in type_ids for token_id in token_ids):
        story = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return story, story, None

    removed_ids = set(eos_ids)
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            removed_ids.add(int(token_id))
    display_ids = [token_id for token_id in token_ids if token_id not in removed_ids]
    raw_text = tokenizer.decode(display_ids, skip_special_tokens=False)
    raw_text = raw_text.strip()

    body = raw_text
    if display_ids:
        first_token = tokenizer.convert_ids_to_tokens(display_ids[0])
        if first_token in tokenizer.all_special_tokens and re.fullmatch(r"\[STORY_TYPE_[1-6]\]", first_token):
            # Hide only the initial structural token; preserve misplaced tokens.
            body = tokenizer.decode(display_ids[1:], skip_special_tokens=False).strip()
    text, names = render_person_tokens(body)
    names = {token: name for token, name in names.items() if token in present_tokens}
    return raw_text, text, names or None
