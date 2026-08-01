# Sensemaking at CLEF 2026: Automatic Evaluation of Answers to Open Questions

**http://ufal.mff.cuni.cz/see/sensemaking-2026**

We focus on the “Evaluator” step, i.e., scoring student responses or, in general, evaluating the quality of answers to open questions based on some input materials.

Despite the impressive fluency and usefulness of modern LLMs, they are easily fooled and can deliver very unreliable evaluations.

We thus seek empirical answers to questions like: Are LLMs indeed capable of making sense of texts? Can LLMs be used to reliably judge answers to questions, grounded in a specific context? And if this holds for the big publicly accessible models, are some of the smaller models comparably capable? Can encoder-only models be equally capable for complex text understanding under our task’s conditions?The Task

## Tracks
The task runs in two tracks:

### Simple Rating

The Simple Rating track format is straightforward: given a triple of texts (context, question, answer), score the answer with the appropriate rating label of 0-4. Here you are trying to match the slightly subjective rating of the reference annotator.

### Rubric-Based Rating

The Rubric-Based Rating track is only slightly more complicated: given a quadruple of texts (context, question, rating rubric, answer), label the answer with either No Credit (NC, label 0), Partial Credit (PC, label 1), or Full Credit (FC, label 2), according to the rubric. The rating rubric (three plain textual descriptions) will consist of three parts, describing when each of the possible labels should be applied.

### Common

In addition the data point will have a language tag to enable using different models for different languages/language groups and a unique id to make submitting your results as simple as giving JSON a list of {"id": str, "label":int, "misc":Any}. Misc here refers to any miscellaneous intermediary results you have obtained. If there are any such results it is mandatory to provide them (ie. if you are using an LLM with reasoning or chain of thought prompting we will consider your submission incomplete if it does not give the misc value for each item, even though we understand sometimes it might be empty or nonsensical). For example the thinking of an LLM, name of the model used for this specific example or the embedding vector used for similarity.

In each track, the metric used to compare models will be the **quadratic weighted kappa**, computed separately for each domain and averaged over domains.


## Multilinguality
The contexts, questions, and answers will be in the following languages:

- English
- Czech
- Portuguese
- German
- Hungarian
- Finnish
- Danish
- Swedish
- Serbian
- Greek
- Irish
- Romanian
- Ukrainian

Depending on the domain, the texts may be originals in the given language, or they may be automatically translated versions. We include only languages for which we have at least one domain with original text.
