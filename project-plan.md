# Can Graph Neural Networks (GNNs) Enable Causal Reasoning in Finance?
Research Project by Alphanome.AI

**Abstract**. In recent years, Graph Neural Networks (GNNs) have emerged as a promising tool for understanding structured data, including financial networks. This research project seeks to explore the potential of GNNs in enabling causal reasoning within the financial sector, assessing both the possibilities and the limitations of the method. Through a series of experiments, models, and real-world financial data, this study will shed light on the efficacy of GNNs in modeling causal relationships and predicting financial outcomes based on these relationships.

**​​Objective:** Enhance Decision-Making with Causal Insights from SEC Filings

Our project aims to develop a sophisticated causal AI framework capable of uncovering and quantifying causal relationships from SEC filings that directly impact company performance metrics, such as revenue growth and profitability. By meticulously parsing through the vast expanse of structured and unstructured data available in 10-K and 10-Q reports, along with other relevant financial disclosures, the framework will identify key financial indicators, management discussion analysis, and market conditions that are causally related to a company's financial outcomes.

We aspire to translate these findings into actionable insights that can inform strategic business decisions, investment choices, and risk assessment strategies. Through the careful application of advanced natural language processing techniques and causal inference algorithms, our approach will be designed to not only establish correlations but also to distinguish causation from mere association, providing a more robust basis for predictions and business intelligence. To accomplish this, the framework will incorporate advanced machine learning techniques, specifically leveraging Graph Neural Networks (GNNs) alongside traditional causal analysis methods. The integration of GNNs aims to capture the complex, interconnected relationships between variables in high-dimensional data that often escape more conventional statistical models.

**This objective will be addressed by achieving the following key goals:**

- Parse and structure data from SEC filings, employing our [open-source sec-parser](https://github.com/alphanome-ai/sec-parser) to extract relevant financial data points and narrative disclosures with high precision by [organizing them into semantic elements and a tree structure.](https://github.com/orgs/alphanome-ai/discussions/18) (Requires testing and improved accuracy) 
- Implement a multifaceted approach to causal analysis that includes both established techniques—such as [Difference-in-Differences (DiD)](https://www.alphanome.ai/post/difference-in-differences-did-analysis-for-investors), [Regression Discontinuity Design (RDD)](https://www.alphanome.ai/post/unlocking-causal-insights-in-investments-with-regression-discontinuity-design-rdd), and [Instrumental Variables (IV)](https://www.alphanome.ai/post/unveiling-causality-a-dive-into-instrumental-variable-analysis-for-investors)—and state-of-the-art machine learning algorithms like [Graph Neural Networks (GNNs)](https://www.alphanome.ai/post/graph-neural-networks-gnn-architectures-and-variants-a-guide-to-the-future-of-data-analysis). These methods, each serving complementary purposes, will be deployed to uncover and validate potential causal links between various financial indicators extracted from SEC filings and the core company performance metrics. The use of GNNs, in particular, will allow our models to learn and represent more complex interactions, providing a deeper and more nuanced understanding of the causal structures that drive a company’s financial outcomes.
- By capitalizing on the rich, graph-structured data inherent within financial filings, our approach seeks to leverage the power of GNNs to create a holistic view of the interconnected elements influencing business performance, paving the way for sophisticated, data-driven decision-making.
- To showcase our findings on a small scale we will develop a dynamic and user-friendly visualization interface that allows stakeholders to easily explore and interpret the causal relationships uncovered, fostering transparency and understanding.
- Validate the reliability and consistency of our causal AI framework by back-testing findings against historical data and performance outcomes, ensuring methodological robustness and practical applicability.
- Continuously refine our causal inference models by integrating feedback from cross-functional teams and keeping abreast of the latest academic and industry research in the fast-evolving field of causal AI.

Our commitment is to bridge the gap between massive data availability and strategic knowledge application, empowering our stakeholders with a cutting-edge tool to make more informed, data-driven decisions that can propel their business toward sustainable growth and enhanced profitability.

# Project Plan

## Data Collection

### Improve `sec-parser` accuracy
- Download a large and diverse set of SEC EDGAR 10-Q filings.
- Parse the filings with [alphanome-ai/sec-parser](https://github.com/alphanome-ai/sec-parser). 
- Create [Github Issues](https://github.com/alphanome-ai/sec-parser/issues) to list any errors found in the parser, then fix [the issues](https://github.com/alphanome-ai/sec-parser/issues).

### Create datasets
- Create a pipeline that includes [sec-parser](https://github.com/alphanome-ai/sec-parser) and other tools to extract data from SEC filings, and elsewhere. Include documentation, examples, tests, and robustness measures.
- Run the pipeline for a set of companies and create a dataset of extracted data.

## Textual Analysis

Perform text analysis on the extracted textual sections to identify topics or segments related to specific financial metrics. This could involve:
- Keyword extraction to identify relevant financial terms.
- Named entity recognition to locate mentions of financial metrics.
- Sentiment analysis to gauge the tone surrounding the discussion of metrics.

Links to some useful resources:
- [Metaculus Presents — Causal Inference and LLMs: A New Frontier](https://www.youtube.com/watch?v=PT1NoaeYwDs)
- [SOTA Entity Recognition English Foundation Model by NuMind](https://huggingface.co/numind/generic-entity_recognition-v1)

## Data Correlation

Identify correlated items (textual or financial) first from individual and then several different 10-Q  documents by using as an assistant LLMs and your own judgment to validate the findings. 

This could involve:
- Creating a mapping of commonly used financial language to the financial metrics (e.g., "revenue growth" to quarterly revenue changes).
- Connect discussions in the text with nearby presented financial figures.
- Management's Discussion and Analysis (MD&A) section: extract sentiments or forward-looking statements that may correlate with financial projections or historical data.
- Identify patterns or frequency of certain risk-related keywords that could correlate with financial outcomes or contingencies.

Develop a method to correlate textual sections with the extracted financial metrics.

## Graph Database Creation

- Use a graph database or a graph construction toolkit to create nodes for each identified financial metric and textual section.
- Create edges based on the correlations found between the text and the metrics. These edges can represent the strength and type of the relationship.

## Visualization

- Choose a visualization tool capable of handling graph-based data.
- Design the visualization to represent different types of connections (e.g., solid lines for direct references, dashed lines for inferred relationships).
- Include interactive features that allow users to click on a node (textual section or financial metric) to see all connected elements and the nature of their connections.

## Interpretation Layer

- Implement an interpretation layer that provides insights when a user explores the connections. This could be a text box displaying excerpts from the report that justify the connection, or a sidebar summarizing the financial trends related to the selected node.

## User Interface

- Develop a user-friendly interface that allows non-technical users to navigate the graph easily.
- Implement search and filter functionalities to let users focus on specific metrics or report sections.

## Validation

- Review the graph and its connections with financial analysts to ensure accuracy.
- Test the system with multiple SEC filings to validate the generalizability of your approach.

## Continuous Learning

- Incorporate machine learning algorithms that can improve the correlation model as more data is processed. This would involve training the system on annotated datasets where the connections between text and financial metrics are pre-labeled.

# Graph Neural Networks

Graph Neural Networks (GNNs) are a class of neural networks designed to perform inference on data described by graphs. They are particularly well-suited for tasks where the data has an inherent graph structure, such as social networks, molecular structures, and in this case, the semantic and relational structure of financial documents.

## Preprocessing Data

- **Node Representation**: Represent each financial metric and textual section as a node in the graph. For textual nodes, embeddings can be generated using pre-trained language models like BERT or GPT. For financial metric nodes, you might use normalized values or embeddings derived from their numerical features and metadata.
- **Edge Representation**: Define the edges by the relationships between text and financial metrics. Initial edges might be defined by co-occurrence or explicit mentions of financial metrics within or near the textual sections.

## Training the GNN Model

- **Graph Construction**: Use the nodes and edges to construct a graph that represents the entire document or corpus of SEC filings.
- **Feature Engineering**: Each node and edge can have features, such as the type of financial metric, the sentiment of the text, or the section of the report it appears in.
- **Labeling**: For supervised learning, you'll need to label parts of your graph to train the GNN. For instance, when a text segment talks about revenue, and there's a revenue figure nearby, that's a positive label for the connection between that text node and the revenue metric node.
- **Model Selection**: Choose an appropriate GNN architecture based on the task. Common models include Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Message Passing Neural Networks (MPNNs).
- **Loss Function**: Define a loss function that captures the correctness of connections. In a supervised setting, this might be a classification or regression loss, depending on whether you're classifying types of relationships or predicting the strength of relationships.

## Model Training

Train the GNN on the labeled graph data. The model learns to propagate and aggregate information across the graph structure, capturing both local and global graph properties.

## Inference and Validation

- After training, use the GNN to infer relationships in new, unseen SEC filings.
- Validate the inferred relationships with experts or against a labeled test set to ensure the model accurately identifies meaningful connections between text and financial metrics

## Visualization and Interaction

- Use the trained GNN to provide real-time predictions and insights. For instance, when users are exploring an SEC filing, the GNN can highlight textual sections most relevant to selected financial metrics.
- Visualize the graph with attention or relevance scores derived from the GNN, highlighting the strength and certainty of each connection.

## Continuous Improvement

- Implement a feedback loop where users can confirm or correct the model’s predictions, providing new labeled data for further training and refinement of the model.
- Explore unsupervised or semi-supervised approaches to further improve the model using the vast amounts of unlabeled data in SEC filings.

Using GNNs can be particularly powerful in this setting because they can learn complex patterns of connections between different parts of the documents, which might not be immediately apparent to human analysts or traditional machine learning techniques. They are especially good at taking into account the contextual information and the structured relationships between different entities within the data.
