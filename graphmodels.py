import matplotlib.pyplot as plt
import numpy as np

# Data
gru_metrics = {
    "Average total errors": 12.00,
    "Errors per 100 words": 5.41,
    "Word count": 224.60,
}

lstm_metrics = {
    "Average total errors": 48.80,
    "Errors per 100 words": 22.73,
    "Word count": 215.80,
}

gru_errors = {
    "TYPOS": 6.80,
    "PUNCTUATION": 3.60,
    "GRAMMAR": 0.80,
    "MISC": 0.40,
    "COLLOCATIONS": 0.20,
    "CASING": 0.20,
}

lstm_errors = {
    "TYPOS": 25.80,
    "PUNCTUATION": 8.60,
    "GRAMMAR": 6.00,
    "CASING": 3.60,
    "TYPOGRAPHY": 2.40,
    "MISC": 1.80,
    "COLLOCATIONS": 0.20,
    "STYLE": 0.20,
    "CONFUSED_WORDS": 0.20,
}

#overall metrics
labels = list(gru_metrics.keys())
gru_vals = list(gru_metrics.values())
lstm_vals = list(lstm_metrics.values())

x = np.arange(len(labels))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, gru_vals, width, label='GRU')
rects2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM')

ax.set_ylabel('Values')
ax.set_title('GRU vs LSTM: Overall Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()

# Annotate bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()

#plot error categories
all_categories = sorted(set(gru_errors.keys()) | set(lstm_errors.keys()))
gru_vals = [gru_errors.get(cat, 0) for cat in all_categories]
lstm_vals = [lstm_errors.get(cat, 0) for cat in all_categories]

x = np.arange(len(all_categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, gru_vals, width, label='GRU')
rects2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM')

ax.set_ylabel('Average Errors')
ax.set_title('Error Categories Comparison: GRU vs LSTM')
ax.set_xticks(x)
ax.set_xticklabels(all_categories, rotation=45, ha="right")
ax.legend()

# Annotate bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

plt.tight_layout()
plt.show()
