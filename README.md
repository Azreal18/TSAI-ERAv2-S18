# TASI_ERAv2_S18

## Objective

Train your transformer (encoder-decoder) in a way such that
1. Loss should be less that 1.8 in 18 epochs
2. Accelerate your transformer training significantly (can use smart batching, One Cycle Policy, Automatic Mixed Precision etc),  

## Dataset - opus_books

- The dataset is from https://huggingface.co/datasets/Helsinki-NLP/opus_books
- This is a collection of copyright free books aligned by Andras Farkas, which are available from http://www.farkastranslations.com/bilingual_books.php
- The dataset supports 16 different languages.
- The dataset is setup as translation between 2 languages. e.g. for English to Italian `{ "en": "\"Jane, I don't like cavillers or questioners; besides, there is something truly forbidding in a child taking up her elders in that manner.", "it": "— Jane, non mi piace di essere interrogata. Sta male, del resto, che una bimba tratti così i suoi superiori." }`
- This experiment uses only the English - Italian pair of sentences, which contains around 32k rows.

##  Model Architecture

## Modified Model Architecture

The transformer model is built with N stacked Encoder-Decoder blocks, each containing multi-head attention mechanisms. In our specific case, the transformer is comprised of 6 Encoder-Decoder blocks. Tokens are embedded into 512-dimensional vectors (d_model). Each block utilizes multi-head attention with 8 heads (h) and is followed by feed-forward networks with a size of 128 (d_ff). The model incorporates positional encodings to capture sequence context and projects the decoder's output to the target vocabulary for translation.

## Model Dimensions

The model has the following dimensions:

- Embedding Dimension (d_model): Determines the size of the embedding vectors. A common choice is 512.
- Feed-Forward Dimension (d_ff): Determines the size of the internal layers in the feed-forward networks present in both the encoder and decoder blocks.
- Number of Attention Heads (h): Influences the number of different attention patterns the model can learn.

## Training Optimization

To enhance the efficiency and performance of the model, the following optimization techniques are employed:

- Parameter Sharing: The weights between the source and target embeddings are shared, reducing the number of parameters and aligning the vector spaces. This is particularly beneficial for tasks involving closely related languages like English and French.

- Automatic Mixed Precision (AMP): This technique speeds up training by utilizing both FP16 and FP32 data types, ensuring minimal loss in model accuracy.

- Dynamic Padding: Sequences in each batch are dynamically padded to the length of the longest sequence in that batch, reducing computational overhead.

- One Cycle Policy (OCP): This learning rate scheduling technique enables faster convergence and potentially better model outcomes.

By combining these optimization techniques, the model trains faster, requires less memory, and achieves improved performance.

##  Results

Final loss after 18 epochs is 1.438, It had hit the critarea of loss under 1.8 by 9th Epoch where the loss was 1.724