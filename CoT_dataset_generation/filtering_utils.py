"""Shared, backend-agnostic logic for the CoT pre-verdict-extraction filtering pipeline.

Used by both the OpenRouter live-call path (filter_CoT_datasets.ipynb) and the Azure
GlobalBatch path (filter_CoT_datasets_azure_batch.ipynb) so the two backends can never
silently drift apart on what counts as a correct extraction point or how a correction
gets applied to a row.
"""

from ast import literal_eval
from typing import Optional

from pydantic import BaseModel
from tqdm import tqdm


SYSTEM_PROMPT = """You are a verifier. You are given (a) the FULL reasoning trace of an LLM deciding whether a factual or arithmetic statement is true or false, and (b) an extracted prefix of that trace that is supposed to end at the pre-final-judgement word. Decide whether the extract ends at the correct point, and if not, return the correct pre-final-judgement phrase.

=== PROCEDURE ===
1. Read the FULL trace top to bottom.
2. Scan for the FIRST whole-statement verdict, in ANY form (see disclosure list). This includes an explicit 'the statement is false / must be false / might be false' that appears EARLY and is then double-checked and restated later — the EARLY one is the disclosure, the restatement is not. A first-pass disclosure that the model then re-checks and CONFIRMS (same answer) is still the disclosure: cut before it, not before the confirming restatement. Only if the re-check FLIPS the answer does the corrected later disclosure become the cut.
3. The correct cut is the token immediately before that first disclosure.
4. The extract is CORRECT only if it ends exactly there. If a disclosure sits inside the extract (including an early verdict the model later restated) → INCORRECT. If it ends on component reasoning before any disclosure → INCORRECT.

=== THE DISCLOSURE CAN TAKE ANY LEXICAL FORM — scan for all of these, not just 'true'/'false' ===
- Conjunction aggregation (VERY COMMON in multi-part statements, often the earliest): 'both parts hold', 'both parts are false', 'both parts are correct', 'both conditions are met', 'both counts are wrong', 'both parts of the statement hold'.
- Reserved-word equivalent: 'matches', 'accurately reflects / accurately counts', 'holds true', 'aligns with', 'doesn't match', 'doesn't hold', 'contradicts the statement', 'there's a discrepancy / a discrepancy'.
- Whole-computation checkout (ARITHMETIC): 'the calculation checks out', 'the whole thing checks out', 'everything checks out' when it refers to the ENTIRE equation. (NOT 'each step checks out' / 'the steps check out' — that is component-level; see NOT-a-disclosure list.)
- Verdict adjective: 'the statement is accurate / inaccurate' — treat 'accurate'/'inaccurate' exactly like 'correct'/'incorrect'.
- Plain-language paraphrase OF THE WHOLE VERDICT: "that's exactly what happened", "that's exactly what the statement/equation says", "just as claimed", "that's not the case", "the opposite of what's true". Counts ONLY when it is about the WHOLE statement/equation matching the claim (e.g. "750+50=800 — that's exactly what the equation says" → the whole equation is true). It does NOT count when it confirms a single step ("that's exactly what I got for 272*24") — that is component-level, see NOT-a-disclosure list.
- Explicit: 'the statement is false', 'the statement is incorrect', 'makes the statement false'.

Multi-part statements are conjunctions ('exactly N in X AND exactly M in Y'). The model often aggregates or equivalent-discloses ('both parts hold', 'accurately counts') one or more sentences BEFORE the explicit 'the statement is/seems to be/must be…' line. When that genuinely happens, the explicit line is a RESTATEMENT and the correct cut is before the earlier disclosure. But an extract ending on 'the statement is/seems to be' is CORRECT whenever the only thing preceding it is facts, counts, per-item checks, or bare computation — do not flag it just because it ends there. Verify what actually precedes it before deciding.

=== COUNT / COMPUTATION RESTATEMENT IS NOT A DISCLOSURE ===
Restating the tallies — 'exactly 3 are in Canada and 3 in Kenya', 'so that's 2 in each' — is the model repeating the FACTS, not committing to the statement's truth, even when it directly precedes 'the statement is/seems to be…'. It is NOT a disclosure.

ARITHMETIC (single-equation) analog — IMPORTANT: these traces have ONE equation, not a two-part conjunction, so the following are bare computation, NOT disclosures and NOT aggregations:
- 'both sides are equal' / 'both sides equal 28' / 'the left side equals the right side' — this is stating the computed equality, the arithmetic equivalent of a restated tally. Do NOT treat it as a 'both parts hold' aggregation. Cut at the verdict that FOLLOWS it, not here.
- 'X, not Y' / 'it's 48, not 43' / '22 isn't equal to 24' / 'doesn't equal' — bare (in)equality of two numbers is computation. (The evaluative comments 'doesn't match' and 'there's a discrepancy' ARE disclosures; the plain numeric '(isn't/doesn't) equal' is not.)
A disclosure needs a truth-value word, an equivalent (matches / doesn't match / holds / accurate / a discrepancy / checks out[whole]), or a paraphrase OF THE VERDICT — never a bare re-statement of the numbers or a bare numeric equality. When these are restated and then 'the statement is …' follows, the explicit line IS the first disclosure and the cut before it is CORRECT.

=== CONJUNCTION RULE (multi-part statements; which sub-part commitments count) ===
- A FALSE sub-part IS a disclosure. One false conjunct settles the whole statement: cut before the first one ('the second part is false', 'which isn't true', 'that part is wrong', 'that part doesn't add up').
- A TRUE sub-part is NOT a disclosure on its own — one true conjunct decides nothing. Do NOT cut at 'the first part holds true' / 'that part checks out'; wait for the aggregation over both parts, or the next part if it is false.
Take whichever disclosure (false sub-part OR aggregation OR equivalent OR explicit) comes FIRST in reading order.

=== WHAT IS NOT A DISCLOSURE (stays inside the extract; never the cut) ===
- Component facts / counts / restated tallies (see above): 'only 1 is in Yemen, not 5', '5 cities in China and 1 in Belarus'.
- Bare computation / numeric equality (ARITHMETIC): 'both sides equal 28', 'it's 48, not 43', '22 isn't equal to 24', 'so the product is 6528'.
- Per-item checkmarks: 'City: Country. Correct.' — the reserved word 'Correct' here confirms ONE city's placement, NOT the whole statement. Likewise 'that's correct' / 'yeah, that's right' confirming a recount of ONE arithmetic step (e.g. '272*24 is 6528, that's correct'). Never cut at these, no matter how close to the end of the extract they appear.
- Component check-out (ARITHMETIC): 'each step checks out', 'the steps check out', 'that step is right' — confirms the arithmetic steps, NOT the statement's truth (a false statement can have every step check out). Only 'the calculation / the whole thing checks out' (whole-equation) is a disclosure.
- A single TRUE sub-part (see conjunction rule).
- Hypothetical / conditional verdicts: 'if X were in Y, the statement would be false' — no commitment.
- The opening framing: 'determine whether this statement is true or false'.
- Any restatement AFTER the first disclosure.

=== LANDING TOKEN ===
- Return the pre-final-judgement word exactly as in the trace, kept even when polarity-bearing ('holds', 'isn't', 'doesn't') — do not move earlier to find a neutral word. When the verdict spans two tokens (polarity + verdict word: "doesn't match", "holds true", "isn't true"), the disclosure word is the SECOND one (match / true), so cut before IT and keep the polarity word as the landing token — e.g. "…which doesn't match" → phrase ends "…which doesn't", NOT before 'doesn't'.
- For an evaluative disclosure led by a subject+copula ('there's a discrepancy', 'that's exactly what the equation says'), cut before the evaluative COMPLEMENT and KEEP the subject+copula — land on 'there's' / 'that's', exclude 'a discrepancy' / 'exactly what the equation says'. This is the same as keeping 'is' in 'the statement is false' and keeping 'that' before 'matches'. Do NOT skip back to an earlier content token (e.g. the computed number) — that breaks consistency with the 'matches' landing.
- Because a bare word like 'is' recurs many times, return it WITH 2–5 words of the immediately preceding context, verbatim, so it locates unambiguously.

=== EXAMPLES ===
Example A (aggregation before explicit → INCORRECT):
  Trace tail: "...Since both parts of the statement hold, the entire statement should be true."
  Extract ends at: "...the entire statement should be"
  -> extraction_point_is_correct: false ; new_pre_final_judgement_phrase: "both parts of the statement"
  (Cut before 'hold'. The 'should be' line is a restatement.)

Example B (equivalent before explicit → INCORRECT):
  Trace tail: "...So the statement accurately counts 2 Kenyan and 2 Polish cities, so it should be true."
  Extract ends at: "...so it should be"
  -> extraction_point_is_correct: false ; new_pre_final_judgement_phrase: "So the statement"
  (Cut before 'accurately'. Landing token 'statement', with preceding context 'So the statement'.)

Example C (counts restated → single verdict, no earlier disclosure → CORRECT):
  Trace tail: "...So exactly 3 are in Canada and 3 in Kenya. The statement seems to be true."
  Extract ends at: "...The statement seems to be"
  -> extraction_point_is_correct: true ; new_pre_final_judgement_phrase: null
  (The restated tally is facts, not a disclosure; the explicit line is the first commitment.)

Example D (false sub-part → INCORRECT, cut there not at the later explicit line):
  Trace tail: "...5 cities are in Colombia, so that part is wrong. Therefore, the entire statement is false."
  Extract ends at: "...Therefore, the entire statement is"
  -> extraction_point_is_correct: false ; new_pre_final_judgement_phrase: "in Colombia, so that part is"
  (One false conjunct settles it; cut before 'wrong', not before the later 'false'.)

Example E (early verdict, restated at end → INCORRECT):
  Trace: "...Therefore, the entire statement is false because 5 are in Colombia. Alternatively, maybe I misread... No. Therefore, the statement is false."
  Extract ends at: "...Therefore, the statement is"
  -> incorrect ; phrase: "in reality there are 5. Therefore, the entire statement"
  (The FIRST 'entire statement is false' is the disclosure; the extract ends at the restatement, too late.)

Example F (TRUE sub-part → extract too early → INCORRECT):
  Trace: "...That makes two cities in Malaysia. So the second part also holds true. [Poland part already checked as 0.] ...the statement is true."
  Extract ends at: "...So the second part also holds"
  -> incorrect ; phrase: <token before the aggregation/explicit line that resolves BOTH parts>
  (One true conjunct is non-determining; wait for the aggregation or the explicit verdict.)

Example G (ARITHMETIC 'doesn't match', confirming re-check after → land on polarity token):
  Trace: "...gives 48, but the statement says 43. That doesn't match. Did I make a mistake? Let me recheck... 48, yes. So the statement is false."
  Extract ends at: "...So the statement is"
  -> incorrect ; phrase: "says 43. That doesn't"
  (First disclosure is 'doesn't match'; keep polarity token 'doesn't', cut before 'match'. The re-check confirms the same answer, so the later 'the statement is false' is a restatement.)

Example H (ARITHMETIC sub-step 'that's correct' is NOT a disclosure):
  Trace: "...272 * 24 is 6528. Yeah, that's correct. But the statement says 6526, which doesn't match, so the statement is false."
  Extract ends at: "...Yeah, that's"
  -> incorrect ; phrase: "says 6526, which doesn't"
  ('that's correct' confirms the sub-product 6528, not the statement. The first whole-statement disclosure is 'doesn't match'.)

Example I (ARITHMETIC 'both sides equal' is computation, not aggregation → CORRECT):
  Trace: "...so both sides equal 28. That means the equation is correct."
  Extract ends at: "...That means the equation is"
  -> extraction_point_is_correct: true ; new_pre_final_judgement_phrase: null
  ('both sides equal 28' is the computed equality for a single equation — bare computation, not a 'both parts hold' aggregation. The explicit line is the first disclosure.)

Example J ('there's a discrepancy' → land on 'there's', exclude the complement):
  Trace: "...the product should be 7168, not 7171. Hmm, there's a discrepancy here. Did I make a mistake? Let me recheck... 7168, yes. Therefore the statement is false."
  Extract ends at: "...Therefore the statement is"
  -> incorrect ; phrase: "Hmm, there's"
  (First disclosure is 'there's a discrepancy'; cut before 'a discrepancy', KEEP 'there's' — same as keeping 'that' before 'matches'. Confirming re-check → the later verdict is a restatement.)

Example K ('that's exactly what … says' → whole-verdict paraphrase, land on 'that's'):
  Trace: "...750 + 50 equals 800. Oh, wait, that's exactly what the equation says! So it does equal 800... the equation is correct."
  Extract ends at: "...the equation is"
  -> incorrect ; phrase: "wait, that's"
  (Whole-equation match paraphrase; keep 'that's', exclude 'exactly what the equation says'. NOT a sub-step 'that's correct'. The later 'the equation is correct' is a restatement.)

=== OUTPUT ===
- extraction_point_is_correct: true only if the extract already ends immediately before the EARLIEST disclosure.
- new_pre_final_judgement_phrase: if incorrect, the correct phrase (word + preceding context); if correct, null."""


def get_statement(generated_statement_text: str) -> str:
    return generated_statement_text.split("<｜User｜>")[1].split("\n")[0]


class StructuredExtractionOutput(BaseModel):
    """Pydantic schema used by the OpenRouter live-call path (client.chat.completions.parse)."""
    extraction_point_is_correct: bool
    new_pre_final_judgement_phrase: Optional[str] = None


# Hand-built strict JSON schema equivalent of StructuredExtractionOutput, for the Azure
# Batch path -- raw JSONL request bodies can't go through .parse()'s automatic Pydantic ->
# schema conversion, so this is kept in sync with StructuredExtractionOutput by hand.
# Strict mode requires every key listed in "required" (Optional means nullable value, not
# an omittable key) and additionalProperties: false.
STRUCTURED_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "StructuredExtractionOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "extraction_point_is_correct": {"type": "boolean"},
                "new_pre_final_judgement_phrase": {"type": ["string", "null"]},
            },
            "required": ["extraction_point_is_correct", "new_pre_final_judgement_phrase"],
            "additionalProperties": False,
        },
    },
}


def _char_to_token_idx(tokenizer, ids, upper_tok, target_char):
    lo, hi = 0, upper_tok
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) <= target_char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def apply_correction(dataset, idx, row, correction, tokenizer) -> bool:
    """Applies one correction dict ({"extraction_point_is_correct", "new_pre_final_judgement_phrase"})
    to one dataset row in place. Returns True if the row was modified.

    Shared by both backends so the phrase-matching/ambiguity-check/truncation logic can
    never drift apart between the OpenRouter live-call path and the Azure Batch path.
    """
    if correction["extraction_point_is_correct"]:
        return False

    phrase = correction["new_pre_final_judgement_phrase"]
    think_open_idx = row["generated_statement_texts"].find("<think>")

    start = row["generated_statement_texts"].find(phrase, think_open_idx)
    if start == -1:
        tqdm.write(f"Row {idx}: phrase not found verbatim, skipping: {phrase!r}")
        return False

    second_occurrence = row["generated_statement_texts"].find(phrase, start + 1)
    if second_occurrence != -1:
        tqdm.write(f"Row {idx}: phrase found multiple times (ambiguous), skipping: {phrase!r}")
        return False

    target_char = start + len(phrase)
    token_idx = _char_to_token_idx(
        tokenizer=tokenizer,
        ids=row["generated_statement_ids"],
        upper_tok=len(row["generated_statement_ids"]),
        target_char=target_char,
    )
    new_ids = row["generated_statement_ids"][:token_idx]

    dataset.loc[idx, "extracted_statement_ids"] = str(new_ids)
    dataset.loc[idx, "extracted_statement_texts"] = tokenizer.decode(new_ids, skip_special_tokens=True)
    return True


def apply_manual_corrections(dataset, corrections: dict, tokenizer, dataset_file_path):
    """corrections: {row_index: phrase}. phrase == "drop" removes the row entirely (e.g.
    truncated/unfinished generations with no genuine pre-verdict activation to extract).

    Reuses apply_correction() for the phrase-application case, so a manual fix goes through
    the exact same phrase-matching/ambiguity-check/truncation logic as an LLM-sourced one --
    never a separate, potentially-drifted implementation.
    """
    if len(dataset) > 0 and isinstance(dataset["generated_statement_ids"].iloc[0], str):
        dataset["generated_statement_ids"] = dataset["generated_statement_ids"].apply(literal_eval)

    to_drop = []
    for idx, phrase in corrections.items():
        if phrase == "drop":
            to_drop.append(idx)
            continue

        correction = {"extraction_point_is_correct": False, "new_pre_final_judgement_phrase": phrase}
        if apply_correction(dataset, idx, dataset.loc[idx], correction, tokenizer):
            tqdm.write(f"Row {idx}: corrected.")

    if to_drop:
        tqdm.write(f"Dropping {len(to_drop)} row(s): {to_drop}")
        dataset = dataset.drop(index=to_drop)

    dataset.to_csv(dataset_file_path, index=False)
    return dataset
