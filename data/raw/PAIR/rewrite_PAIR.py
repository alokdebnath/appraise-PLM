import pandas as pd

df = pd.read_csv('pair_data.csv')

prompts = []
responses = []
quality = []

for index, row in df.iterrows():
    for col in df.columns:
        if 'hq' in col:
            prompts.append(row['prompt'])
            responses.append(row[col])
            quality.append(3)
        elif 'mq' in col:
            prompts.append(row['prompt'])
            responses.append(row[col])
            quality.append(2)
        elif 'lq' in col:
            prompts.append(row['prompt'])
            responses.append(row[col])
            quality.append(1)

# Create a new DataFrame with the extracted data
data = pd.DataFrame({
    'prompt': prompts,
    'response': responses,
    'quality': quality
})

clean_df = pd.DataFrame(data)
clean_df.to_csv('clean_pair_data.csv', index=False)