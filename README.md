# Clustering Analysis of Knee Osteoarthritis Structural Progression Profiles
## Motivation and rationale  

### The Context:  
Osteoarthritis is a common condition which affects joints creating pain and difficulty in moving, in this I will be focusing on occurrences within the knee joint. It is caused by cartilage breaking down within the joints which can occur from injuries, family history and obesity [1]. Osteo is the most common form of arthritis affecting over 32.5 million adults in the US alone [2]. It is a common condition in the older population because it occurs due to usage and so increasing age means an increased risked of osteoarthritis.  

### The Problem:  
Many subtypes of diseases are present within the osteoarthritis cohort but knowing distinct subtypes could further our understanding of the disease. With this furthered understanding we can realise cases where people may be more predisposed to osteoarthritis than initially expected. As well as this we can improve treatment plans to be more relevant to specific groups and narrow research areas. The main difficulty will be clustering the data from the progression profiles as there are many different algorithms all with different variations and parameter options. So I will need to measure the success of the clusters produced.

### My Approach:
For my project I intend to analyse patient progression profiles from a knee osteoarthritis cohort  and find similarities in these profiles, showing distinct subtypes of diseases. To do this I will unify the data from these profiles doing and use a clustering algorithm to group data by similarities. These clusters will show the mentioned subtypes of diseases and how common they are. To make this information understandable I will produce visualisations of the findings from this.  
I aim to do this with the use of Python and its libraries of Pandas and Seaborn, for data analysis and producing graphs respectively. I am using these libraries as I have some experience with both and Seaborn is designed Panda’s data structures.  

## Aim and objectives  
### Aim: Analyse data from patient progression profiles to find common subtypes of diseases and visualise the findings
### Objectives:
1)	Understand the data present in the progression profiles  
Improving my understanding of what the data in these profiles represents and means will allow me to refine my research areas and ability to analyse the data. I will complete this objective before I begin to unify the data as otherwise mistakes will likely occur.  
2)	Unify the data from progression profiles  
Combining all of the data from the profiles means I can begin implementing a clustering algorithm. It will also give me a new view of the data from which I can hopefully begin to draw some conclusions and predictions of what the clusters may resemble when produced, enabling me to make informed decisions when assessing the clusters.  
3)	Explore different clustering algorithms that can be used with the data  
These algorithms will group similar data together showing common occurrences between the progression profiles and trends in the data. I will visualise these clusters because the clustering algorithms have various parameters each effecting the clusters produced and in turn the provided view, so I need to know how they differ visually from one another.  
4)	Group common similarities between profiles using optimal clustering algorithm  
I will discover the optimal clustering algorithm using metrics built-in to scikit-learn [4]. For example, using “Silhouette Coefficient” which produces a numerical score representing the distance between clusters, where the aim is to achieve a high score.  This will allow me to quickly assess the clustering algorithms and tune them as necessary.  
5)	Visualise the similarities in an informative manner  
With the data being clustered I can begin to draw conclusions from the groupings by producing visualisations of the data. These visualisations should show distinct subtypes of diseases.  
## Background  
| Source                             | Description | Relevance |
|------------------------------------|-------------|-----------|
| (Unifying data) [4]                |             |           |
| Types of Clustering Algorithms [5] | This discusses four types of clustering algorithms that can be used, all of which contain their own variants but the focus will be on these types included. Each variant produces very different visuals of clusters showing how different these types are and the need to explore different types rather than variants. | Clustering will be a big part of the project, I will produce many variations of clusters using clustering algorithms all with various parameters. These are the most common algorithms and will be the focus of my experiments in clustering. |
| Clustering with Scikit-learn [3]   | Scikit-learns documentation provides explanations of the workings of multiple variations of clustering algorithms and how they work. As well as methods of assessing the performance of said algorithms. | This provides easy implementation of some of the algorithms to be explored and methods of testing their performance using standard methods. |
| (High Quality Visualisations) [6]  |             |           |

## Work plan  
### Done so far: (Will be done)
*	Background research on unifying data
*	Background research on clustering algorithms
*	Workplan for future
### Future Plans:
*	Unify the data
*	Implement and test clustering algorithms
*	Produce visualisations

#### (Insert Ganttchart)

## References  
[1] NHS (2019). Overview - Osteoarthritis. [online] NHS. Available at: https://www.nhs.uk/conditions/osteoarthritis/.  
[2] CDC (2020). Osteoarthritis (OA). [online] Centers for Disease Control and Prevention. Available at: https://www.cdc.gov/arthritis/basics/osteoarthritis.htm.   
[3] Scikit-learn.org. (2010). 2.3. Clustering — scikit-learn 0.20.3 documentation. [online] Available at: https://scikit-learn.org/stable/modules/clustering.html.  
[5] Google Developers. (2015). Clustering Algorithms | Clustering in Machine Learning. [online] Available at: https://developers.google.com/machine-learning/clustering/clustering-algorithms.  
