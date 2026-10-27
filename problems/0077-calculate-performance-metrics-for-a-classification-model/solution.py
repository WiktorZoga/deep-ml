import  numpy as np

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:

	confusion_matrix = np.zeros(shape=(2, 2))

	for a, p in zip(actual, predicted):
		confusion_matrix[1 - a, 1 - p] += 1

	tp = confusion_matrix[0, 0]
	fn = confusion_matrix[0, 1]
	fp = confusion_matrix[1, 0]
	tn = confusion_matrix[1, 1]

	n = len(actual)

	accuracy = (tp + tn) / n
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	specificity = tn / (tn + fp)
	negativePredictive = tn / (tn + fn)
	f1 = 2 * (precision * recall) / (precision + recall)

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
