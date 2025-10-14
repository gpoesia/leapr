import logging
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

@cache
def load_llm(model: str):
    if '/' not in model:
        raise ValueError("Model name must be in the format 'provider/model_name' (e.g. openai/gpt-5-mini)")

    provider, model_name = model.split("/", 1)

    # We have to do this because the generic langchain interface doesn't play well
    # with multiprocessing (there's no multiprocessing we use that involves calling LLMs in parallel,
    # this is just related to Python trying to replicate global imports, so sillier than that).
    if provider == 'openai':
        model_class_ = ChatOpenAI
    elif provider == 'anthropic':
        model_class_ = ChatAnthropic
    else:
        raise ValueError(f'No handler for provider {provider}')

    return model_class_(model=model_name)


def generate_features(
    model: str,
    prompt: str,
) -> list[str]:
    """Generate new chess features using an LLM."""
    print('Prompt:')
    print('###' * 20)
    print(prompt)
    print('###' * 20)

    logger.info(f"Generating discriminant chess features using model {model}")

    try:
        llm = load_llm(model)
        response = llm.invoke([('user', prompt)])

        content = response.content

        print('Raw response:')
        print('###' * 20)
        print(content)
        print('###' * 20)

        # Parse the generated features
        assert content, "LLM response is empty"
        features = _parse_features(content)

        logger.info(f"Generated {len(features)} features")
        return features

    except Exception as e:
        logger.error(f"Error generating features: {e}")
        return []



def _parse_features(content: str) -> list[str]:
    """Parse feature functions from LLM response."""
    features = []
    current_feature = ""
    capturing = False

    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Start capturing when we see def feature
        if line.startswith("def feature("):
            if capturing and current_feature.strip():
                features.append(current_feature)
            current_feature = lines[i] + "\n"
            capturing = True

        # Stop capturing at various end markers
        elif capturing and (
            line.startswith("```")
            or line == "### END"
            or line.startswith("def feature(")
            or (line.startswith("#") and "explanation" in line.lower())
            or line.startswith("These features")
        ):
            if current_feature.strip():
                features.append(current_feature)
            current_feature = ""
            capturing = False

            # If this line starts a new feature, handle it
            if line.startswith("def feature("):
                current_feature = lines[i] + "\n"
                capturing = True

        # Continue capturing
        elif capturing:
            current_feature += lines[i] + "\n"

        i += 1

    # Add the last feature if there is one
    if capturing and current_feature.strip():
        features.append(current_feature)

    return features
