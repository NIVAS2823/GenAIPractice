from rag_queryexp_chain import answer_with_query_expansion

if __name__ == "__main__":
    q = input("Ask(with Query Expansion + re ranking):")
    answer,docs = answer_with_query_expansion(q)


    print("\n===FINAL ANSWER===")
    print(answer)

    print("===Top Matched Chunks==")
    for i,d in enumerate(docs,start=1):
        print(f"\n{i} {d.page_content[:200]}\n---")