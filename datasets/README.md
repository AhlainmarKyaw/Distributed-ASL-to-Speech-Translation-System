# Dataset placement

## Alphabet
Place a public ASL alphabet image dataset here in class folders:

datasets/alphabet_images/A/*.jpg
datasets/alphabet_images/B/*.jpg
...
datasets/alphabet_images/Z/*.jpg

The extractor ignores non A-Z folders such as `space`, `del`, or `nothing`.

Recommended source: Kaggle ASL Alphabet by Akash Nagaraj / grassknoted.
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Note: J and Z are motion signs. A static-image classifier can be weaker for them. The phrase/dynamic sequence pipeline is the correct place to extend motion recognition.

## Phrases / dynamic signs
The phrase trainer expects 30-frame MediaPipe landmark `.npy` sequences under:
datasets/phrases/HELLO/
datasets/phrases/THANK_YOU/
...

You can create these with `training/collect_phrase_sequences.py`, or preprocess appropriately licensed public word-level ASL video datasets such as WLASL into the same 30x63 format.
