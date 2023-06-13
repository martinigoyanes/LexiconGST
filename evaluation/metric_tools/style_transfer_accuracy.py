from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
from torch.nn.utils.rnn import pad_sequence


def classify_preds(args, preds, task: str, device="cpu"):
    assert task
    print('Calculating style of predictions')
    results = []

    if task == 'jigsaw' or task == 'paradetox':
        # 0 -> neutral, 1 -> toxic
        tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
    if task == 'yelp':
        # 0 -> negative, 1 -> positive
        path = '/Midgard/home/martinig/thesis-src/models/roberta_sentiment_classifier'
        tokenizer = RobertaTokenizer.from_pretrained(f"{path}/tokenizer")
        model = RobertaForSequenceClassification.from_pretrained(f"{path}/model")

    model.to(device)
    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        batch.to(device)
        result = model(**batch)['logits'].argmax(1).float().data.tolist()
        if task == 'jigsaw' or task == 'paradetox':
            # Sums 1 if text is neutral
            results.extend([1 - item for item in result])
        if task == 'yelp':
            # Sums 1 if text is positive
            results.extend([item for item in result])

    return results