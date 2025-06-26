from dotenv import load_dotenv
import dspy
from typing import Literal

if __name__ == '__main__':
    load_dotenv()
    lm = dspy.LM("openai/gpt-4o-mini", max_tokens=3000, cache=False)
    dspy.configure(lm=lm)

    sentence = "its a charming and often affecting journey."

    classify = dspy.Predict('sentence: str -> sentiment: bool')
    result = classify(sentence=sentence)
    print(f"Sentiment is {result.sentiment}")

    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

    summarize = dspy.ChainOfThought("document -> summary")
    response = summarize(document=document)

    print(f"Response is {response}")
    print("Reasoning:", response.reasoning)

    class Emotion(dspy.Signature):
        """classify Emotion"""
        sentence: str = dspy.InputField()
        sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()


    sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"
    classify = dspy.Predict(Emotion)
    print(f'The emotion is {classify(sentence=sentence)}')

    class CheckCitationFaithfulness(dspy.Signature):
        """Verify that the text is based on the provided context"""
        context: str = dspy.InputField(desc="Facts here are assumed to be true")
        text: str = dspy.InputField()
        faithfulness: bool = dspy.OutputField()
        evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")


    context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
    text = "Lee scored 3 goals for Colchester United."

    faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    print(f'Faithfulness check: {faithfulness(context=context, text=text)}')


    class DogPictureSignature(dspy.Signature):
        """Output the dog breed of the dog in the image"""
        image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
        answer: str = dspy.OutputField(desc="The breed of the dog in the image")


    image_url = "https://picsum.photos/id/237/200/300"
    classify = dspy.Predict(DogPictureSignature)
    breed = classify(image_1=dspy.Image.from_url(image_url))
    print(f"Dog breed is {breed.answer}")

    print(f"Length of LM usage history is {len(lm.history)}")

    print(f"Cost for {lm.history[-1]['prompt']} is {lm.history[-1]['cost']}")
