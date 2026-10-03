# Transformer (Model 5)

Figures are in `results/transformer/figures/`. All numbers come from `transformer-models.ipynb`.

## Related Work

Transformers use self-attention instead of recurrence, so each token can look directly at every other token in the sequence (Vaswani et al., 2017). For text classification, most of their strength comes from pre-training on large amounts of text and then fine-tuning on the task (Devlin et al., 2019; Sun et al., 2019). Smaller distilled versions such as DistilBERT keep most of the accuracy (Sanh et al., 2019). For tweets, models pre-trained on Twitter data do better than general models on sentiment and stance tasks (Nguyen et al., 2020; Barbieri et al., 2020), which is an example of domain-adaptive pre-training (Gururangan et al., 2020). Fine-tuning on small datasets is also known to be sensitive to the random seed and learning rate (Dodge et al., 2020; Mosbach et al., 2021). Based on this we compared a Transformer trained from scratch, general pre-training (DistilBERT) and tweet pre-training (BERTweet), and we ran a learning rate sweep and several seeds.

## Dataset and Exploratory Analysis

- Tweets have 16.3 words on average, which becomes 27.6 WordPiece tokens (DistilBERT) and 23.8 BPE tokens (BERTweet). A `max_len` of 48 keeps more than 99.5% of tweets whole.
- 10.8% of test words never appear in the training data, and 72.7% of test tweets have at least one such word. This hurts word-level models like the CNN and LSTM but not subword models. DistilBERT splits a hashtag into 4.1 pieces on average (`#whatcausesautism` becomes `# what ##ca ##uses ##au ##tism`), BERTweet into 3.2.
- Negation words appear in 32.4% of Negative and 33.0% of Positive tweets, so their presence alone says nothing about stance. The model has to understand what is being negated, so we kept stop words.
- Only 36.6% of Negative training tweets have full annotator agreement, compared with 64.3% for Neutral and 58.2% for Positive. The smallest class is also the noisiest.

## Methodology

Preprocessing: HTML entities are fixed and `<user>`/`<url>` are replaced with `@USER`/`HTTPURL`, the tokens BERTweet was trained with. Case, punctuation, emojis and stop words are kept. We used the group's shared 70/15/15 split.

| ID | Setup | Purpose |
|---|---|---|
| T1 | 2-layer Transformer encoder from scratch (d=128, 4 heads, 4.2M params) | no pre-training |
| T2 | DistilBERT-base-uncased (67M) | general pre-training |
| T3 | BERTweet-base (135M) | tweet pre-training |
| T4 | T3 + class-weighted loss | class imbalance |
| T5 | T4 + agreement as sample weight | noisy labels |
| T6 | learning rate 1e-5, 2e-5, 3e-5, 5e-5 | tuning |
| T7 | best setup with 3 seeds | variance |

Training used AdamW (Loshchilov & Hutter, 2019) with weight decay 0.01, 10% warm-up then linear decay, batch size 32, gradient clipping 1.0 and up to 5 epochs, with early stopping on validation macro-F1 (patience 2). All decisions were made on the validation set.

Metrics: macro-F1 is the main metric because each class counts equally, which matters with only 10% Negative tweets (Rosenthal et al., 2017; Barbieri et al., 2020). We also report Negative recall, macro ROC-AUC, per-class average precision (Saito & Rehmsmeier, 2015), MCC (Chicco & Jurman, 2020) and RMSE of the expected score, which was the Zindi metric.

## Results and Discussion

| Model (test set, n = 1500) | Macro-F1 | Accuracy | Neg. recall | ROC-AUC |
|---|---|---|---|---|
| T1 Scratch Transformer | 0.632 | 0.725 | 0.31 | 0.840 |
| T2 DistilBERT | 0.692 | 0.765 | 0.46 | 0.885 |
| T3 BERTweet (selected) | 0.751 | 0.797 | 0.58 | 0.909 |
| T5 BERTweet + class and agreement weights | 0.751 | 0.789 | 0.66 | 0.912 |
| T3, 3 seeds (mean ± std) | 0.735 ± 0.014 | 0.789 ± 0.007 | 0.56 ± 0.02 | 0.906 ± 0.003 |

The Transformer trained from scratch scored about the same as the SVM, CNN and LSTM (0.62 to 0.64) and started overfitting after epoch 4. With only 7k tweets, self-attention alone has no advantage, because it doesn't have the built-in assumptions that help CNNs (local patterns) and LSTMs (word order) learn from less data. General pre-training added about 0.06 macro-F1.

Pre-training on tweets added about another 0.06, with the biggest gain on Negative tweets. This matches the EDA finding that BERTweet handles hashtags and Twitter tokens better, and the results of Gururangan et al. (2020). BERTweet is also twice the size of DistilBERT, so part of the gain may come from size.

Class weights, agreement weights and the learning rate sweep changed validation macro-F1 by 0.01 or less, while the seed standard deviation was 0.014, so these differences are within noise. A second full run showed the same thing with a different winner (weighted model with LR 3e-5, test 0.756, seeds 0.747 ± 0.010). The one consistent effect is that class and agreement weights raise Negative recall from 0.58 to 0.66 without lowering macro-F1. In the CNN and LSTM notebook, class weights changed results much more. The pre-trained model already separates the classes well (ROC-AUC about 0.91), so the weights only move the threshold a little.

Compared with the other models, the Transformer improves macro-F1 by 0.09 to 0.13 and nearly doubles Negative recall, but it needs a GPU (about 9 minutes per run) and is much larger.

## Error Analysis and Limitations

- Most mistakes are on tweets where annotators disagreed: 8.0% error when all agreed and 37.0% when they didn't (macro-F1 0.872 vs 0.604). These tweets are 42% of the test set but 77% of the errors.
- The most common mistake is Neutral→Positive (132 tweets), mostly health news that uses the same words as pro-vaccine tweets. Many confident Negative→Neutral mistakes are news headlines with agreement 0.33, which look mislabelled.
- Sarcasm is still hard. "Got a vaccine yesterday so now I'm part of the conspiracy" (labelled Positive) was predicted Negative because of the words "vaccine" and "conspiracy".
- Negation has only a small effect: 22.6% error with negation vs 19.6% without.
- The model is overconfident (ECE about 0.12). At 85–95% confidence it is right only about 64% of the time, so it would need calibration (Guo et al., 2017).
- Limitations: one split, 3 seeds and only the learning rate tuned. With 156 Negative test tweets, one tweet is about 0.6 points of recall. Domain and model size are mixed in the DistilBERT vs BERTweet comparison.
- Responsible AI: the model was trained on English tweets from one period and has not been tested on African languages, code-switching or newer topics like COVID-19 vaccines. It should only be used for overall trends, not to target individuals, and people should review its outputs.

## Future Work

Calibration with temperature scaling, training on soft labels from annotator votes, COVID-Twitter-BERT (Müller et al., 2023), a RoBERTa-base comparison for model size, and combining the Transformer with the SVM.

## References

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of EMNLP 2020*, 1644–1650.
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics, 21*, 6.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*, 4171–4186.
- Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. *arXiv:2002.06305*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
- Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't stop pretraining: Adapt language models to domains and tasks. *ACL 2020*, 8342–8360.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines. *ICLR 2021*.
- Müller, M., Salathé, M., & Kummervold, P. E. (2023). COVID-Twitter-BERT: A natural language processing model to analyse COVID-19 content on Twitter. *Frontiers in Artificial Intelligence, 6*, 1023281.
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP 2020: System Demonstrations*, 9–14.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). SemEval-2017 Task 4: Sentiment analysis in Twitter. *SemEval-2017*, 502–518.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? *CCL 2019*, 194–206.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020: System Demonstrations*, 38–45.
- Pre-trained models: `vinai/bertweet-base`, `distilbert-base-uncased` (Hugging Face Hub). Libraries: PyTorch, Hugging Face Transformers, scikit-learn.
