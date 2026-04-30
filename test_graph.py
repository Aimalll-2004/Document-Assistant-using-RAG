from graph import app

questions = [
    "How are machine learning models trained?",
    "What are transformers used for?",
    "What is pizza dough made of?"
]

for q in questions:
    result = app.invoke({
        "question": q,
        "route": "",
        "result": "",
        "answer": ""
    })

    print("\nQuestion:", q)
    print("Route:", result["route"])
    print("Result:", result["result"])
    print("Answer:", result["answer"])










