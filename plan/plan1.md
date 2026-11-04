I'll write the five files first, then lay out the steps.

The scenario is a small café chain in Kigali, **Inzozi Café**, with a helper that answers questions from its own files (all invented data). I've attached the five XML files, and all of them parse as valid XML.

| File | What it holds |
|---|---|
| `menu.xml` | 7 items with price, vegan flag and allergens |
| `branches.xml` | 3 branches with opening hours, plus holiday rules |
| `leave-policy.xml` | Staff leave rules by length of service |
| `refund-policy.xml` | 7 refund rules (RF-1 to RF-7) |
| `delivery.xml` | Delivery zones, fees and times |

Staff and customers can ask questions, and the answer has to say which file and which element it came from.

**Why XML helps you.** In a PDF you cut text by size and cite a page. Here the file already has natural pieces: an `<item>`, a `<rule>`, a `<branch>`, and each has an `id`. So a chunk is one element, and a citation is `file + element id`, for example `refund-policy.xml · RF-3`. That is simpler and more exact than pages.

## Your build order, with a check at every step

| Step | You build | Done when |
|---|---|---|
| **0. Questions** | Write 12 questions and the file and id of each answer (my 12 are below). | You have the list. This is your answer key. |
| **1. Models** | Tables for the source file and the chunk (file, element id, text, embedding, model name). | A migration runs up and down. |
| **2. Parse and chunk** | Read each XML file and make one chunk per `item`, `rule`, `branch` or `zone`. Add a breadcrumb to the text, for example "Menu > Drinks > Masala Chai: ...". | You get the right count (about 30 chunks), and each one has a file and an id. |
| **3. Embed** | A function that turns a list of texts into vectors. | The same text gives the same vector, and the size matches your column. |
| **4. Store** | Insert the chunks, and make re-running safe (delete and re-insert per file). | Running the import twice leaves the same row count. |
| **5. Query** | Embed the question and take the 3 nearest chunks. | Run all 12 questions and count how many have the right chunk in the top 3. This is your **hit rate**. |
| **6. Answer** | Send the question and the 3 chunks to the model, and ask for an answer plus `file` and `element id`. Check in code that the id was one of the chunks you sent. | Wrong ids are dropped, and the unanswerable questions say "not found". |
| **7. Tokens** | Log input and output tokens per question, and compare with sending all five files. | You have a number for "RAG used X% fewer tokens". |
| **8. Improve** | One change at a time, re-measure the hit rate after each. | See below. |

I read "drug search" as **hybrid search** (keyword search plus vector search), so that's what the list below assumes. For step 8, the order I'd try is:
1. Hybrid search (keyword plus vector, merged by rank).
2. Tuning how many chunks you take (top 3 vs top 5).
3. A re-ranker.
4. Rewriting follow-up questions.

**Tips for parsing:** use Python's `xml.etree`, but for XML you don't control, use `defusedxml`. XML can carry tricks (huge nested entities) that crash a parser.

## Your 12 test questions

| # | Question | Answer is in |
|---|---|---|
| 1 | Is the Masala Chai vegan? | `menu.xml` D-02 (no, milk) |
| 2 | What time does Kimihurura close on Sunday? | `branches.xml` B-KIM (18:00) |
| 3 | How many days of annual leave after 2 years? | `leave-policy.xml` L-1.2 (18) |
| 4 | Can a customer get a refund on a drink that was already served? | `refund-policy.xml` RF-1 |
| 5 | How much is delivery to Nyamirambo? | `delivery.xml` DL-B (2,500) |
| 6 | Which items contain peanuts? | `menu.xml` F-04 only |
| 7 | Is Kacyiru open on Saturday? | `branches.xml` B-KAC (closed) |
| 8 | How long do refunds take to arrive? | `refund-policy.xml` RF-5 (5 working days) |
| 9 | I ordered 8 days ago, can I get a refund? | `refund-policy.xml` RF-7 (no) |
| 10 | What does rule RF-3 say? | `refund-policy.xml` RF-3 (an exact-id question) |
| 11 | What is the price of a cappuccino? | **Not in the files.** The answer must be "not found" |
| 12 | Is there parking at Kimihurura? | **Not in the files.** The answer must be "not found" |

These cases are planted on purpose. #10 is where vector search tends to fail because it needs an exact word, so hybrid search should fix it. #11 and #12 test that the model refuses instead of guessing. #6 needs several chunks, and #9 needs the model to reason about a rule.

Before you write any code, make your own guess for step 5: for which of the 12 questions do you expect the right chunk to **not** be in the top 3? Write it down, then compare with the real hit rate once step 5 works.

If you'd like the files placed in your repo (for example under `data/inzozi/`), tell me the folder.