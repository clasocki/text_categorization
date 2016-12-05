from semantic_model import DocumentIterator
from collections import Counter

if __name__ == "__main__":
	word_count = Counter()

	for document_batch in DocumentIterator().getAllInBatches(cond="published = 1 AND journal_id = 8356"):
		for document in document_batch:
			print document.journal.title
			for word in document.tokenized_text:
				word_count[word] += 1

	for word, count in word_count.most_common(100):
		print word, count