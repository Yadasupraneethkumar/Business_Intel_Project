import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()


def get_groq_client() -> Groq:
    """
    Create and return a Groq client.

    Returns:
        Groq client.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return Groq(api_key=api_key)


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:
    """
    Generate a structured response using Groq.

    Args:
        prompt: Input prompt.
        response_model: Pydantic model describing the expected output.
        model: Groq model name.

    Returns:
        Parsed Pydantic response model.
    """

    model = model or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    client = get_groq_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert financial analyst. "
                "Extract information accurately from the provided "
                "financial report context. "
                "Return only the requested structured information."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # ---------------------------------------------------------
    # Primary approach: Groq Structured Outputs
    # ---------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema()
                }
            }
        )

        text = response.choices[0].message.content

        if not text:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        print("[debug] Groq structured output:")
        print(text)

        parsed = response_model.model_validate_json(text)

        return parsed

    # ---------------------------------------------------------
    # Fallback: JSON Object Mode
    # ---------------------------------------------------------
    except Exception as exc:

        print(
            f"[debug] Structured output failed: {exc}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert financial analyst. "
                            "Return ONLY valid JSON. "
                            "Do not include markdown, explanations, "
                            "or code fences."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_object"
                }
            )

            text = response.choices[0].message.content

            if not text:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            print("[debug] Groq JSON fallback:")
            print(text)

            # Remove markdown code fences if the model
            # accidentally includes them.
            text = text.strip()

            if text.startswith("```"):
                text = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    text,
                    flags=re.IGNORECASE
                )
                text = re.sub(
                    r"\s*```$",
                    "",
                    text
                )

            # Extract JSON object if additional text exists.
            match = re.search(
                r"\{.*\}",
                text,
                re.DOTALL
            )

            json_text = (
                match.group(0)
                if match
                else text
            )

            data = json.loads(json_text)

            return response_model.model_validate(data)

        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to obtain a valid structured response "
                f"from Groq.\n"
                f"Structured-output error: {exc}\n"
                f"Fallback error: {fallback_exc}"
            ) from fallback_exc

