# coding=utf-8
mrk_title =  '''
## A visual exploration of nearly 10,000 tweets about Clemson Football

This brief demo uses AI tools including IBM Watson to create a visual representation of a sample of tweets
about Clemson Football collected in September and October 2018. 
Human concepts of _topic_ (like Kelly Bryant's transfer), _sentiment_ (like positivity and negativity), 
and _emotion_ (like anger) manifest visually by their color and position in the 3D space. The visuals 
and accompanying discussions which follow introduce you to the notion of exploring human communication
in space and demonstrate a few unique insights brought to light by this visual, interactive approach. 

Most of the visuals are interactive, so go ahead and experiment!  
'''

intro_title =  '''
### Exploring text in 3D
'''

intro_descr = '''
**What am I looking at?**

The graph on the left shows nearly 10,000 tweets matching the Twitter search "Clemson Football" collected 
over a few weeks. Each point corresponds to a single
tweet. You can read the tweet by hovering over the point. You can also
zoom, pan, and tilt with the mouse (go ahead and try).

**But what does the position in space mean?**

Posts are "embedded" into a 3D space based on their word usage, topic, sentiment, and emotions. 
Posts that are similar across these dimensions will appear close to one another in space.

**What are the colors?**

The colors represent clusters of topically related posts. These clusters were determined using an _unsupervised_
machine learning algorithm. You can usually figure out what the overall topic is for a given color by reading a few
posts. For example, the orange points near the bottom of the view are all talking about 
Trevor Lawrence as starting QB. 
'''

sentiment_title =  '''
### Post Sentiment
'''

sentiment_description = '''
Sentiment describes the overall negativity or positivity of a post on a range from negative to neutral to positive 
(red to grey to blue in visuals). These sentiment scores were generated using the 
[Watson Natural Language Understanding](https://www.ibm.com/watson/services/natural-language-understanding/) service.

In the scatterplot on the right, we show the post's sentiment as color -- red is negative, blue is positive. 
Most clusters, have almost entirely positive, negative, 
or neutral posts. Also, within most clusters, the degree of sentiment will vary as you move from one side of a 
cluster to the other. A few clusters even show a range of perspectives on 
a given topic as a range of colors (for example, people posting about their experience of watching
Clemson play on TV).

By dragging over the histogram below, you can select posts within a range of sentiment values. Using this tool,
you can explore the most positive or negative posts to identify patterns. In this data set, there is a great deal of 
negativity surrounding a certain game against Syracuse and events related to Kelly Bryant. 
'''

emotion_title = '''
### Post Emotion
'''

emotion_description = '''
In addition to sentiment, Watson identifies the emotional content of the posts, in particular,
_anger, disgust, fear, joy,_ and _sadness_. These give a richer description of the posts than the content alone. 

In the following figures, the points are colored by their extremity
along each emotional dimension. The slider above the scatterplot allows you to select the coloring
emotion, and the histograms on the left allow you to select the range of each emotion to show on the right. The combination
of these elements is a rich environment for understanding the patterns in the data.  

For instance, is it possible to tweet in a way that is both joyful and angry? By using the histograms to select both
joyful and angry posts, we find 

> @coachfostervt @HokiesFB @VT_Football defense ?? Pass rush ?? Db play ? It’s ODU not Clemson. Come on. PPW. Play, pride ,WIN. Come on.

which is about as close as you can get.
'''
