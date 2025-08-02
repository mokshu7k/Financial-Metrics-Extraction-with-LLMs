# Financial Metrics Extraction and Analysis with Generative AI  
This project explores using Generative AI to automate financial analysis in two parts. It begins with a foundational research paper and builds on those findings to create an intelligent, agent-based system for financial data extraction.  

This is not just a simple script; it is a complete workflow that evaluates different AI methods and uses the best ones to build a powerful application.  

Part 1: The Research  
The first phase of this project looked into a key question: How accurate and efficient are different Generative AI methods at extracting specific financial numbers, like Revenue or Net Income, from long, unstructured earnings call transcripts?  

I implemented and tested three core techniques:  

Zero-shot: The simplest approach, where the model extracts data based on a direct prompt.  

Chain-of-Thought (CoT): A method that prompts the model to "think step by step," enhancing its reasoning.  

Retrieval-Augmented Generation (RAG): A technique that gives the model relevant context from the transcript, which significantly improves accuracy.  

The findings from this research are documented in a formal paper, which is also included in this repository.  

Part 2: The Agentic System  
Using insights from my research, particularly that RAG is the most reliable method, I developed an Agentic AI system. This is not just a tool that follows instructions; it is an intelligent agent capable of reasoning and making decisions.   

This agent acts like a junior financial analyst. You can ask it a complex question like, "What was Company X's revenue last quarter, and what is its current P/E ratio?" The agent then autonomously performs these steps:  

Plans: It decides which tools it needs to use to answer the question.  

Uses a Tool: It calls the custom RAG Extractor tool to find the revenue number in the transcript.  

Uses another Tool: It calls a Web Search tool to find the current P/E ratio.  

Synthesizes: It combines information from both sources and provides a complete answer.  

This multi-step reasoning makes the agent a powerful and flexible tool for financial analysis.  

Why This Project Matters  
This project shows a full-cycle development process: from foundational research to building a practical, intelligent application. It demonstrates a deep understanding of key Generative AI concepts such as prompting techniques, RAG, and Agentic AI. This is a strong example of how these technologies can work together to address real-world problems in finance and other data-driven fields.