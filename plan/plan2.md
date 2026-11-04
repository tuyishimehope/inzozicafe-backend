Yes. Your foundation is now solid: chunking, embeddings, hybrid search, citations, and evals that measure everything. Here's what to add, in the order I'd build it. Each item fixes a problem you've already seen or will soon. Keep the rule you already follow: one change, then re-run the scorecard.

## 1. A "not found" gate *(fixes a problem you've seen)*

Search always returns 3 chunks, even for the cappuccino question. Right now only the model protects you.
- **Add:** if the best chunk's cosine distance is above a threshold, skip the model and answer "not found".
- **Learn:** how to pick the threshold *from data*. Print the distances for HITs and for #11/#12, then put the line between the two groups.
- **Bonus:** it saves a chat call on every unanswerable question.

## 2. Re-ranking *(the item you parked)*

Hybrid finds the right chunk **somewhere** in the top 10; a re-ranker puts it **first**.
- **Add:** take the top 10 from hybrid, score each (question, chunk) pair with a cross-encoder, and keep the top 3. Try `BAAI/bge-reranker` locally, or Cohere/Jina rerank via API.
- **Picture:** hybrid is the librarian who brings 10 likely books; the re-ranker actually reads each one against your question.
- **Measure:** add **MRR** (where the right chunk ranks: 1st = 1.0, 2nd = 0.5, 3rd = 0.33) alongside the hit rate. Hit@3 can't see the difference between 1st and 3rd; MRR can.

## 3. Query rewriting *(fixes multi-part and vague questions)*

Questions like #15 ("D-02 and F-04") or "Is it open?" (open *where*?) search badly as one string.
- **Add:** a small LLM call that splits or expands the question, e.g. "What do D-02 and F-04 say?" → two searches, then merge the results.
- **Trade-off:** one extra call per question, which you can now measure with your token script.

## 4. Conversation memory *(makes it feel like a real assistant)*

Follow-ups break: "Is Kacyiru open Saturday?" then "What about Sunday?". The second question alone has no branch name.
- **Add:** rewrite each follow-up into a standalone question using the chat history, *before* searching.
- **Eval:** add 2–3 two-turn cases.

## 5. Streaming answers

- **Add:** a `/ask/stream` endpoint with Server-Sent Events, so the answer appears word by word.
- **Catch:** you can only check citations after the full answer arrives. Think about how to handle that.

## 6. Production habits *(what separates a demo from a system)*

- **A GIN index** on a stored `tsvector` column: keyword search stays fast at 100k chunks.
- **An HNSW index** on `embedding` for vector search at scale.
- **Re-indexing:** when `menu.xml` changes, replace only that document's chunks (hash the file contents).
- **Caching** embeddings of repeated questions, which also helps with your rate limit.
- **Evals in CI:** run `evals.run_retrieval retrieval` on every PR, and fail the build if the hit rate drops.

## 7. LLM-as-judge *(when `must_contain` gets too crude)*

`"no" in answer` passes "I do**n**'t k**no**w". A second model can grade *meaning*: "Does this answer say the Masala Chai is not vegan because it has milk? yes/no". Keep `must_contain` as a cheap first check.

## My recommendation

Do **1 → 2** next. Both fix problems you've already seen in your own output, both are measurable with what you've built, and together they're the core of most production RAG systems.

**Before you start #1, answer this:** why is a distance threshold risky if you later change the embedding model? (Hint: think back to the "two different maps" idea.)