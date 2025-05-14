class BERTimbauVectorizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased').to(device)
        self.model.eval()

    def _get_embeddings(self, texts, batch_size=8):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch.tolist(), return_tensors='pt', 
                                 padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def fit_transform(self, X_train):
        return self._get_embeddings(X_train)

    def transform(self, X_test):
        return self._get_embeddings(X_test)