# Why each generation exists

This project changes one central idea per generation, then evaluates on the same locked ticket IDs. The synthetic sample is intentionally too small to establish that a more complex generation is better.

## V1 → V2: counts to embeddings

V1 assigns each token an independent count. It cannot treat `wifi` and `wireless` as related unless both have useful class counts. V2 learns a vector for each training word by predicting nearby words (skip-gram with negative sampling), then averages those vectors for a ticket. Inspect `src/embeddings/word2vec.py` and run `python -m src.embeddings.inspect vpn` after training. Unknown words still have no vector. Word neighbors from 32 short tickets are mostly noise.

## V2 → V3: order

An average of word vectors is unchanged when the same words are reordered. V3 reads embeddings sequentially with an LSTM; its hidden and cell states can encode earlier context. Inspect `src/baselines/lstm.py`. A tiny dataset can still make it memorize training phrases rather than learn transferable word order.

## V3 → V4: focus

V3 classifies from the last real LSTM output. V4 computes a learned score for **each** LSTM output, masks padding, normalizes scores with softmax, and uses a weighted sum. The model can emphasize words relevant to the label even if they occur early in the ticket. An attention weight is a model operation, not proof that a word caused the decision.

## V4 → V5: self-attention and position

V4 still processes words in sequence through the LSTM. V5 projects every token into a query, key, and value. A query scores every key; softmax weights combine the values. Several heads perform this operation in parallel, followed by a feed-forward block. Because attention alone has no order signal, `src/transformer/positional_encoding.py` adds sinusoidal position vectors before the encoder. Read `src/transformer/attention.py` next to `src/transformer/encoder.py` to trace the tensors.

## V5 → V6: pretraining

V5 learns all its language representations from the small ticket training set. V6 loads a pretrained BERT encoder and freezes it. It uses the first token's representation for each training ticket, computes a centroid per label, and classifies by cosine similarity. The pretrained model may know more language, but the centroids still need representative labeled tickets.

## V6 → V7: task adaptation

V6 never updates the encoder. V7 adds a classification head and updates pretrained weights on the ticket training rows. The head starts with new weights; a small or narrow dataset can make fine-tuning worse. The sample V7 result is an example of that risk, not a sign that fine-tuning is generally inferior.

## V7 → V8: serving

V8 does not change the classifier. It loads a chosen artifact once, validates input length, requires an API key by default, and exposes health and prediction routes. It can serve V1–V7. Production mode checks a private approval manifest against the model and real-data validation and test reports before startup. Follow the [real-ticket evaluation and model promotion workflow](real_data_and_promotion.md) to select a model, then set up secrets and TLS, operational monitoring, and a rollback path.
