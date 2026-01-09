import numpy as np

y = np.load("dataset_processed/words/y.npy")

unique_labels = sorted(set(y))
print("Unique numeric labels:", unique_labels)
print("Total classes:", len(unique_labels))

from collections import Counter
counts = Counter(y)

print("\nSamples per label:")
for k in sorted(counts):
    print(f"Label {k}: {counts[k]} samples")
