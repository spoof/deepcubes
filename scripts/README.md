# Services, scripts and experiment section

Just several examples of usage

## Vera Live Dialog API

### /train

`GET` or `POST` query with `config` field as json string representaion:
```
{
	"lang" (string) possible values: "rus" or "eng"
	"not_understand_label" (string) any string that corresponds to `not understand` label
	"labels_settings": [
		{
			"label" (string) unique label name
			"patterns" (list[string]) regexp with patterns, example: ["нет", "нет.*"]
			"generics": (list[string]) list with generics names ("yes", "no", "repeat")
			"intent_phrases": (list[string]) list with intent questoins phrases
		},
		...
	]
}
```

return json string with `model_id` (int).

### /predict

`GET` or `POST` query with `model_id` (int) field (returned by `/train`) and `query` (string) field as user input text.

return collection of labels sorted decreasingly according probabilities

```
[
	{
		"label" (string),
		"proba" (float)
	},
	...
]
```


## Embedder service

### /get_vector


`GET` query with `mode` (string) ("rus" or "eng") and `tokens` (list[string]) fields.

return json string with `vector` as list of floats with embedding vector components.
