# find where the text is stores for the episodes

import pickle

episode_number = 1

with open(f'data/peract_dataset/close_jar/all_variations/episodes/episode{episode_number}/variation_descriptions.pkl', 'rb') as f:
    description = pickle.load(f)

with open(f'data/peract_dataset/close_jar/all_variations/episodes/episode{episode_number}/variation_number.pkl', 'rb') as f:
    variation_number = pickle.load(f)

print(description)
print(variation_number)