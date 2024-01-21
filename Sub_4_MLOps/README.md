# Submission 4: Exercise ML Ops
Link to repository: https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/tree/main/Sub_4_MLOps 

## Questions to MLOps:
Website: https://ml-ops.org/


1. What are the key principles of ml-ops?
Machine Learning Model Operationalization Management (MLOps) is a term that describes the entire process from the data or issue to the monitoring of the end product. The idea is to design and supervise the entire life cycle from data acquisition to user-oriented optimization of an ML-product. Three main points are data (acquisition and preparation), training and production of a model (ML), integration of the model into the finished product or digital environment.

2. What is model governance in the context of ml-ops and what would be the key points if you explain this to a CEO? 

When it comes to machine learning products, there is a big difference between what is possible and what is legal and feasible. Many companies only realize this when the product is close to completion and fail to bring their product to market. That's why it's important to think about issues such as legal restrictions, user data protection, the dangers of the product, customer safety, IT security and many more right from the start.
Therefore, we need to consider the risks of a product at every stage of development and make risk management an integral part of the product's entire life cycle. This approach is also referred to as "model governance" and describes the complete strategy of risk management.
As software products and ML products in particular are constantly being developed further - the machines and their developers are continuously learning, so to speak - risk minimization through model governance is never complete. 

3. As we had a CI/CD lecture: what is the connection between ml-ops and CI/CD?!
Due to the fact that software products are never fully developed, their further development and improvement must be considered with every product. This is particularly the case with ML products, as they sometimes only develop their strength through constant improvements and over time. They are not only optimized through their application (debugging) but also produce data themselves and thus resources from which they are made or built.
Two important terms from Machine Learning Model Operationalization Management (MLOps) are therefore Continuous Integration (CI) and Continuous Delivery (CD). CI refers to the testing of product parts, from the data (input) to the code (machine) all the way to the finished model (output). CD describes the continuous learning of the model. The model acquires new data, which allows it to improve and optimize itself even if no changes are made to the actual architecture. This process should also be automated, i.e. the training process of the model should be able to be repeated without major human supervision or restructuring. 

4. Describe the MLOps infrastructure stack in two paragraphs!
The long process from the idea to the finished ML product can be roughly boiled down into two steps

§1 Ideate and build the product
After an idea has led to a use case for machine learning (ML product), a prototype can be built. Not only subsequent models are expected, but the entire process (life cycle) is optimized for repetition and automated as far as possible. This ranges from the data set to the code (ML model) to the deployment (pipeline and orchestration e.g. K8) of the product. 
It should be designed in such a way that it continues to run even when the product is released, regarding the fact that a software and ML product is never complete.

§2 Create and operate the product
The ML product must now be manufactured and managed as a service. This second step is also constantly repeated and improved, as the product changes and so does its manufacture and provision or operation.
The resources (data) are processed into a product (ML model) and put into action (released) after evaluation. Now the running product must be supported (operating) and monitored and maintained. This again generates data (resources) that lead to an improved product if the cycle is repeated.