import openai

openai.api_key = "API_KEY"

models = openai.Model.list()
for model in models['data']:
    print(model['id'])
