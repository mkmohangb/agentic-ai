import dspy
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    lm = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)

    output = lm("What is Apache Kafka?")
    print(f"Answer to question is:  {output}")

    print(f"Length of LM usage history is {len(lm.history)}")

    usage_info = lm.history[-1]

    for key in usage_info.keys():
        print(f"{key} is {usage_info[key]}")
