# Data Notes

This project uses the Hugging Face `amazon_polarity` dataset.

- Dataset: `amazon_polarity`
- Source: Hugging Face Datasets library
- Purpose: binary product-review sentiment classification
- Labels: 1 = positive, 0 = negative

The project does not train on the full multi-million-row dataset.
It uses a reproducible subset for laptop-friendly experimentation.

If the dataset cannot be downloaded automatically, follow the install instructions in the main README and run the training script again. The project does not create fake data.
