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
## Background Research
| Source                                            | Description | Relevance |
|---------------------------------------------------|-------------|----------------------------------------------------------------------------|
| Types of Clustering Algorithms [4]                | This discusses four types of clustering algorithms that can be used, all of which contain their own variants but the focus will be on these types included. Each variant produces very different visuals of clusters showing how different these types are and the need to explore different types rather than variants.                                                                | Clustering will be a big part of the project, I will produce many variations of clusters using clustering algorithms all with various parameters. These are the most common algorithms and will be the focus of my experiments in clustering. |
| Clustering Algorithms: A Comparative Approach [5] | This paper gives background information to clustering algorithms and assess’ how common types perform. It shows how performance can vary between the types of data tested and the parameters of each algorithm type. It makes three assessments for each type tested, which are using the default parameters and single or random parameter changes.                            | I need to assess various types of clustering algorithms all with their own parameters, this provides ideas on how to vary the parameters for testing the algorithms quickly and uniformly. It also outlines some common clustering methods and gives some explanations about how they work and what their parameters represent, this information will be useful when implementing them later on.                            |
| What Makes a Visualization Memorable [6]                                    | This discusses the importance of visualisations and how they can be easily insufficient in conveying their meaning. It shows measurements of memorability for various visualizations, showing how effective they are.| I will have to visualise my findings and clusters in meaningful way, in which I mean one where the data can be easily understood. Each of the clustering algorithms will produce different visuals so I need to ensure that when presenting these findings they stand out from one another.                                                                                     |

## Work plan  
### Done so far:
Firstly I have done background research on a few of the topics I will be encountering throughout. The main one being based around clustering algorithms, these will be a big part of the project and are apart of objectives 3 and 4. I have read an article from Google Developers which summarised some of the common types and visualised some example results. This article discussed when each type is better suited depending on the data and how much their produced clusters can vary.  

Furthermore, I have looked at a paper discussing the performances of different types [5], which gave some in-depth comparisons and experiments of common clustering algorithms. This solidified some of my understanding produced from the prior article, as well as explaining some of the workings of the algorithms and how their parameters can affect their results. As a result I have a better idea of where I will start when implementing these algorithms.  

I have decided upon a package to use for implementing the clustering algorithms, Scikit-learn [3], I chose this because it is easy to use and I have some experience with using it in the past. This package contains implementations of clustering algorithms and universal metrics that I can use to assess the clusters produced. Which will make deciding the optimal clustering algorithm much easier.  

As well as this I have decided upon other packages I will be using with Python, they will be Pandas and Seaborn. This is because I have experience with both and they are well suited to the task, Pandas will be used for processing the data and preparing it for implementing the algorithms. Then Seaborn will be used to visualise my findings, I chose this package because it is designed around Pandas and so I know they will work well together making the visualisation stage easier.

### Future Plans:
To start with I will go through the data and familiarise myself with it to ensure I have a basic understanding of what is being shown in each dataset. This will be crucial for the next steps because if I lack this understanding I will be very prone to making mistakes which will delay my progress.
 I will do this by… And I expect to spend so long on it…

*	Unify the data: How will it be done, how long will it take?
*	Implement and test clustering algorithms: What types, how long will it take?
*	Produce visualisations: What will they show and how long will it take?

#### (Insert Ganttchart)

## References  
[1] NHS (2019). Overview - Osteoarthritis. NHS. Available at: https://www.nhs.uk/conditions/osteoarthritis/.  
[2] CDC (2020). Osteoarthritis (OA).  Centers for Disease Control and Prevention. Available at: https://www.cdc.gov/arthritis/basics/osteoarthritis.htm.   
[3] Scikit-learn.org. (2010). 2.3. Clustering — scikit-learn 0.20.3 documentation.  Available at: https://scikit-learn.org/stable/modules/clustering.html.   
[4] Google Developers. (2015). Clustering Algorithms | Clustering in Machine Learning.  Available at: https://developers.google.com/machine-learning/clustering/clustering-algorithms.   
[5] Rodriguez, M.Z., Comin, C.H., Casanova, D., Bruno, O.M., Amancio, D.R., Costa, L. da F. and Rodrigues, F.A. (2019). Clustering algorithms: A comparative approach.  Available at: https://doi.org/10.1371/journal.pone.0210236  
[6] Borkin, M.A., Vo, A.A., Bylinskii, Z., Isola, P., Sunkavalli, S., Oliva, A. and Pfister, H. (2013). What Makes a Visualization Memorable? IEEE Transactions on Visualization and Computer Graphics. Available at: https://ieeexplore.ieee.org/document/6634103   
