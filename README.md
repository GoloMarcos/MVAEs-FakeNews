# One-Class Learning for Fake News Detection Through Multimodal Variational Autoencoders
- Marcos Gôlo (ICMC/USP) | marcosgolo@usp.br (corresponding author)
- Mariana Caravanti de Souza (ICMC/USP) | mariana.caravanti@usp.br
- Rafael Rossi (UFMS) | rafael.g.rossi@ufms.br
- Solange Oliveira Rezende (ICMC/USP) | solange@icmc.usp.br
- Bruno Magalhães Nogueira (UFMS) | bruno@facom.ufms.br
- Ricardo Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Abstract
Machine learning methods to detect fake news are employed with the goal of decreasing disinformation. These models typically use textual features and Binary or Multi-class classification. However, accurately labeling a large news set is still a very costly process. On the other hand, one of the prominent approaches to detect fake news is One-Class Learning. One-Class Learning requires only the labeling of fake news documents, minimizing data labeling effort. Although we eliminate the need to label non-interest news, the efficiency of One-Class Learning algorithms depends directly on the data representation model adopted. Most existing methods in the One-Class Learning literature explore representations based on a single modality to detect fake news. However, different text features can be the reason for the news to be fake, such as topic, misspellings or excessive adjectives, adverbs, and superlatives. We model this behavior as different modalities for news data to represent different sets of textual features. Thus, this paper presents the MVAE-FakeNews, a multimodal method to represent the texts in the fake news detection domain through One-Class Learning, that learns a new representation from the combination of promising modalities for news data: text embeddings, topic information, and linguistic information. In the experimental evaluation, we used real-world fake news datasets considering Portuguese and English languages. Results show that the MVAE-FakeNews obtained a better F1-Score and AUC-ROC for the class of interest, outperforming another fourteen methods in three datasets and getting competitive results on the other three datasets. Moreover, our MVAE-FakeNews with only 3% of labeled fake news obtained comparable or higher results than other methods, i.e., obtaining competitive results in scenarios with few labeled fake news. To improve the experimental evaluation, we also propose the Multimodal Local Interpretable Model-Agnostic Explanations (LIME) for One-Class Learning to identify how each modality is associated with the fake news class.

# Proposal: Triple Variational Autoencoder
![Proposal](/images/proposal.png)

# Github Organization
- Pipfiles: contains the versions of the libraries used
- DEMO: ...
- Results: total results of each method in each scenario considering all parameters used
- Codes: source codes used for the study experiments (including Multimodal LIME for OCL)

