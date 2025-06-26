from dotenv import load_dotenv
import dspy

if __name__ == "__main__":
    load_dotenv()
    lm = dspy.LM("openai/gpt-4o-mini", max_tokens=3000, cache=False)
    dspy.configure(lm=lm)

    question = "What's something great about the ColBERT retrieval model?"

    # pass configuration keys when declaring a module
    classify = dspy.ChainOfThought('question -> answer', n=5)
    response = classify(question=question)

    print(response.completions.answer)

    def evaluate_math(expression: str) -> float:
        return dspy.PythonInterpreter({}).execute(expression)

    def search_wikipedia(query: str) -> str:
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
        return [x['text'] for x in results]

    react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])
    pred = react(question="What is the year of birth of David Gregory of Kinnairdy castle?")
    print(pred.answer)
    pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
    print(pred.answer)
