# AsrSimulator
This software simulates the output of an ASR system using as input plain text.

## Dependencies

The dependencies for this tool cna be found in the requierements.txt file:

nltk==3.4

numpy==1.16.2

singledispatch==3.4.0.3

six==1.12.0


## Example of usage

```python
sentence = "Maybe what I think Tastee Wheat tasted like actually tasted like oatmeal or tuna fish"

asrSimulator_noise = AsrSimulator()
asrText = asrSimulator_noise.convertSentenceToAsrFormat(sentence)

print(sentence) #returns 'Maybe what I think Tastee Wheat tasted like actually tasted like oatmeal or tuna fish'
print(asrText) #returns '['{"Maybe":1.0},{"Maybe"},{0,0.28}', '{"what":1.0},{"what"},{0.38,0.19}', '{"I":1.0},{"I"},{0.65,0.02}', '{"think":1.0},{"think"},{0.74,0.29}', '{"Tastee":1.0},{"Tastee"},{1.14,0.42}', '{"Wheat":1.0},{"Wheat"},{1.63,0.21}', '{"tasted":1.0},{"tasted"},{1.9,0.26}', '{"like":1.0},{"like"},{2.24,0.17}', '{"actually":1.0},{"actually"},{2.48,0.49}', '{"tasted":1.0},{"tasted"},{3.04,0.47}', '{"like":1.0},{"like"},{3.59,0.23}', '{"oatmeal":1.0},{"oatmeal"},{3.94,0.33}', '{"or":1.0},{"or"},{4.35,0.13}', '{"tuna":1.0},{"tuna"},{4.53,0.24}', '{"fish":1.0},{"fish"},{4.84,0.21}']'
```

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)