import os
import sys

from dotenv import load_dotenv
from groq import Groq, GroqError

# The chunk passed to this prompt is made of one or more labeled sources, each
# starting with a header like "[Source N: file_path:start_line]" (see ask.py /
# main.py). The model is told to cite that same N inline so answers stay
# traceable back to a specific chunk, while the "I don't know" fallback is kept
# verbatim so ungrounded questions still get a clean, exact refusal.
SYSTEM_PROMPT = (
    "Answer the question using only the information in the provided chunk of "
    "text. The chunk is made up of one or more sources, each starting with a "
    "header like \"[Source N: file_path:start_line]\". When your answer uses "
    "information from a source, cite it inline like \"(Source N)\", using the "
    "same N from that source's header. If the chunk does not contain the "
    "answer, respond with exactly: I don't know."
)


def query_stream(chunk, question, api_key):
    """Like query(), but yields the answer token by token as Groq streams it back.

    stream=True makes create() return an iterator of chunk events instead of
    one response; each event's delta.content is the next bit of text (or None
    on events that carry no new text, e.g. the final one), so we skip those.
    """
    client = Groq(api_key=api_key)

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Chunk:\n{chunk}\n\nQuestion:\n{question}"},
        ],
        stream=True,
    )

    for event in stream:
        token = event.choices[0].delta.content
        if token:
            yield token


def query(chunk, question, api_key):
    # Same external signature/behavior as before (still returns the full string) —
    # safe to rebuild on top of query_stream since the only difference is that the
    # underlying Groq request is now made in streaming mode instead of one-shot;
    # the final joined text is identical, and this way there's one place (above)
    # that builds the Groq request instead of two.
    return "".join(query_stream(chunk, question, api_key))


def load_api_key():
    """Load GROQ_API_KEY from .env, or print an error and exit."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("Error: GROQ_API_KEY is missing or empty. Set it in your .env file.", file=sys.stderr)
        sys.exit(1)

    return api_key


def safe_query(chunk, question, api_key):
    """Call query(), printing a clean error and exiting if the Groq request fails."""
    try:
        return query(chunk, question, api_key)
    except GroqError as e:
        print(f"Error: request to Groq failed: {e}", file=sys.stderr)
        sys.exit(1)


def safe_query_stream(chunk, question, api_key):
    """Like safe_query, but for query_stream: yields tokens, printing a clean
    error and exiting if the Groq request fails. CLI-only (see ask.py) — a
    long-running server must not let sys.exit reach it, so main.py calls
    query_stream directly instead of this."""
    try:
        yield from query_stream(chunk, question, api_key)
    except GroqError as e:
        print(f"\nError: request to Groq failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python query.py <chunk> <question>", file=sys.stderr)
        sys.exit(1)

    api_key = load_api_key()
    chunk_arg, question_arg = sys.argv[1], sys.argv[2]

    print(safe_query(chunk_arg, question_arg, api_key))
