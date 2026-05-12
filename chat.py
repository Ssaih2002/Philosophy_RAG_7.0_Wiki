from src.rag_engine import RAGEngine

rag = RAGEngine()

print("\nPhilosophy Research Assistant Ready.\n(Type 'exit' to quit)")

while True:
    q = input("\nQuestion: ")
    if q.lower() in ["exit", "quit"]:
        break

    answer, docs, meta = rag.answer(q)
    if meta.get("keywords_used"):
        print("\n检索关键词:", ", ".join(meta["keywords_used"]))
    print("\nAnswer:\n")
    print(answer)