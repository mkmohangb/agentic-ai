import dspy
from dotenv import load_dotenv

load_dotenv()  #GEMINI API key
lm = dspy.LM(model="gemini/gemini-2.0-flash")
dspy.configure(lm = lm)

sign = dspy.Signature("picture -> weight", instructions = "extract the weight value from the image")
weight_extractor = dspy.ChainOfThought(sign)
img = dspy.Image.from_file("weight.jpg")
result = weight_extractor(picture=img)
print(result)
