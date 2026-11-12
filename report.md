Exploring My City’s Sentimental Proportions through Text Classification.

Upon encountering logistic regression classifiers, I was reminded of a metaphor a teacher once taught in class: in a painting filled with shades and darkness, a little light draws the most attention, as the contrast highlights its brightness. In texts dominated by one strong sentiment, one may easily overlook those hidden in plain sight, whether they are blessings disguised as throwaway lines or melancholic descriptions. This leads me to explore the binary problem between positive and negative sentiments, and how the proportions of the two, the rare instances of certain sentiments, may reflect a text’s true nature.

To begin, I defined positive and negative sentiments. Positive sentiments positively impact the text; they can be whimsical descriptions that lift the mood, or comfort and contentment that progress the story. By contrast, negative sentiments negatively impact the text; they can be grim depictions of hard-hitting reality, or disappointment and disillusionment that set a character back. Despite the broadness and ambiguity of the two labels, they can still provide general directions for understanding a text’s emotional fluctuations.

To train the model, I downloaded datasets from qhchina-data/datasets/classification. It is a JSON list of Douban reviews, with 100,000 reviews total, 20,000 reviews per star rating. After labelling 5-star and 4-star reviews as positive (label 1), and 2-star and 1-star reviews as negative (label 0), and dropping 3-star reviews, we’re left with 40,000 reviews per label, which is perfectly balanced. From them, 80% of each label is randomly picked for training, while the remaining 20% of each is used for testing. Random_state=42 was used. Each review is then segmented with Jieba, turned into TF-IDF vectors with sklearn TfidfVectorizer, and fitted on the training reviews only. The logistic regression classifier from scikit-learn is then trained on these reviews, which constitutes the basis for further testing and analysis.

A point worth noting is that Douban book reviews were initially considered for training the classifier, especially those of My City by Xixi, which will be used in later testing. Yet, the idea was ultimately abandoned due to technical difficulties in extracting them into a unified JSON list similar to that from qhchina.

The test-set results are mixed. Statistically, the classifier should be accurate and competent at estimating texts’ sentiments. With the confusion matrix below, the following results can be obtained: Accuracy: 86.66%, Precision: 85.17%, Recall: 88.76%, F1: 86.93%.

Confusion matrix:
Predicted positive, actually positive: 7101
Predicted positive, actually negative: 1236
Predicted negative, actually negative: 6764
Predicted negative, actually positive: 899

However, both labels have misclassified items, unfortunately. Within the respective labels’ top 20 most weighted words, a lot of them should’ve remained neutral and are clearly misclassified, such as “有时”, “很”, “最” from the positive label, and “女主”, “作者”, “实在” from the negative label, to name a few. An interesting pattern of misclassified subjects can be found in the results, as neutral subjects such as “我们”, “女主”, “作者” and “郭敬明” are classified into their respective labels. This may potentially cause overestimation when the classifier is applied, which would be demonstrated later.

To further test the classifier, I chose Xixi’s (西西) novel My City (《我城》) as a separate text for classification. My City is arguably one of Xixi’s most iconic novels, and by extension, one of Hong Kong’s. The story loosely follows the protagonist, 阿果, and his family and friends, and paints a group portrait of local grassroots amid the ups and downs of 1970s Hong Kong.

Over the years, Xixi has garnered a reputation for childlike curiosity in her writing. Like some newcomers to Hong Kong literature, I found myself with two cliché assumptions when first approaching the novel. Before reading the novel, I mistook it for a solely gritty and realistic encapsulation of the socio-political turmoil back then, as local authors have historically paired the word “city” (城) with other adjectives or aliases to metaphorise Hong Kong’s social and cultural zeitgeist. Meanwhile, I found myself doubtful of the story’s point of view, as I first perceived Xixi’s writing style as overly naive and optimistic.

These two assumptions are polarising, to say the least. Yet with the testing results from the trained text classifier, as well as prior experience reading the text, it can be concluded that the truth lies somewhere in between. On one hand, the text undeniably embodies a certain hopefulness; however, it never shies away from heavy subject matter. 

According to the testing results, positive passages took up a staggering 97% of My City, while only 6 passages were estimated as negative. There’s a vast disparity between the two, yet negativity was never avoided in the novel, as concluded.

However, this result is nowhere near perfect. Within positive passages, the sentiments flicker immensely. Meanwhile, when we take a closer look at the estimated passages, the classifier never seems to truly capture the two labels in the novel. At best, the classifier highlighted neutral words, yet the words that truly home in on the positives and negatives, such as “天佑我城” and “美丽” for the former, and “默然道别" and “石油枯竭” for the latter, were never registered.

Looking back at how the classifier was trained, I believe the problem stems from the very limited dataset used. As mentioned in the notes from Week 9’s seminar, the dataset consists of Douban film reviews. The length of each review is quite short as well, contrary to some of the longer reviews one might find on the platform. This may result in its limited glossary in estimating sentiments in the novel, especially in classifying more subtle underlying emotions. Perhaps if book reviews or textbooks were used as datasets for training, the classifier would then label sentiments more accurately.

Regardless, I still find the result yielded from the classifier somewhat a reflection of Xixi’s My City. The proportions between positive and negative sentiments are not balanced in some specific novels, yet to generalise them as swinging to one side would be too crude a conclusion. To circle back to the metaphor from the beginning, only the contrast may highlight the purest sentiments, as reality is always muddled and there is always more running beneath the eye.


-End-
