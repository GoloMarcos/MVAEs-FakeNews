# One-Class Learning for Fake News Detection Through Multimodal Variational Autoencoders
- Marcos Paulo Silva Gôlo (ICMC/USP) | marcosgolo@usp.br (corresponding author)
- Mariana Caravanti de Souza (ICMC/USP) | mariana.caravanti@usp.br
- Rafael Geraldeli Rossi (UFMS) | rafael.g.rossi@ufms.br
- Solange Oliveira Rezende (ICMC/USP) | solange@icmc.usp.br
- Bruno Magalhães Nogueira (UFMS) | bruno@facom.ufms.br
- Ricardo Marcondes Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Abstract
Machine learning methods to detect fake news typically use textual features and Binary or Multi-class classification. However, accurately labeling a large news set is still a very costly process. On the other hand, one of the prominent approaches is One-Class Learning (OCL). OCL requires only the labeling of fake news, minimizing data labeling effort. Although we eliminate the need to label non-interest news, the efficiency of OCL algorithms depends directly on the data representation model adopted. Most existing methods in the OCL literature explore representations based on one modality to detect fake news. However, different text features can be the reason for the news to be fake, such as topic or linguistic features. We model this behavior as different modalities for news to represent different textual features sets. Thus, we present the MVAE-FakeNews, a multimodal method to represent the texts in the fake news detection through OCL, that learns a new representation from the combination of promising modalities for news data: text embeddings, topic and linguistic information. In the experimental evaluation, we used real-world fake news datasets considering Portuguese and English languages. Results show that MVAE-FakeNews obtained a better F1-Score and AUC-ROC, outperforming another fourteen methods in three datasets and getting competitive results on the other three. Moreover, our MVAE-FakeNews with only 3% of labeled fake news obtained comparable or higher results than other methods. To improve the experimental evaluation, we also propose the Multimodal LIME for OCL to identify how each modality is associated with the fake news class.

# Proposal: Triple Variational Autoencoder
![Proposal](/images/proposal.png)

# Github Organization
- Pipfiles: contains the versions of the libraries used
- DEMO: ...
- Results: total results of each method in each scenario considering all parameters used
- Codes: source codes used for the study experiments (including Multimodal LIME for OCL)

