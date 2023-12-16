#### Team Name - Legalysis
#### Problem Statement - AI-Enhanced Legal Practice Platform
#### Team Leader Email - tummuri.hari1@wipro.com

#### A Brief of the Prototype:
      To proficiently address these challenges, we’ve utilized diverse capabilities within large language models (LLMs). Our solution encompasses tasks such as summarizing legal documents, researching past cases, recalling IPC sections, and developing litigation strategies. Leveraging Intel’s robust hardware and the powerful Intel OneAPI AI analytics toolkit, we’ve fine-tuned and optimized LLMs on extensive datasets, culminating in a comprehensive and efficient solution. Alongside fine-tuning, we incorporated Retrieval-Augmented Generation to enhance our approach for the mentioned use cases.

#### Prototype Description:
  To develop this solution, we have gathered data from diverse sources, which consists of legal case judgments, legal proceedings, and the Indian penal code. These     datasets underwent pre-processing, and along with large language model capabilities we utilized various methodologies to devise effective solutions.

  For summarizing the legal documents, we have trained LLAMA2 7B model with legal document summarization dataset resulting in performance improvement of LLM for        legal case summarization. And we have done finetuning using Intel AI analytics toolkit.

  For Legal research, we’ve integrated LLM with RAG (Retrieval Augmented Generation) to access details about specific cases of interest to users. This integration      facilitates the provision of valuable insights for their case proceedings.

  For IPC related queries, we have fine-tuned a T5 large model using an IPC dataset and used Intel developer cloud for finetuning. This enables the model to identify   the specific IPC section applicable to a given scenario.

  ![Architecture](https://github.com/hari-tummuri/oneAPI-GenAI-Hackathon-2023/assets/104126503/4919b2c9-9ea3-43a0-8bcb-c1bfd2dd74b4)

![Process Flow Diagram](https://github.com/hari-tummuri/oneAPI-GenAI-Hackathon-2023/assets/104126503/7678e04b-7c72-42ff-a353-3656a448d636)

  

  
  
  
#### Step-by-Step Code Execution Instructions:
  
1.  Login to IDC(Intel developer cloud)
2.  Install node,express,ngrok
3.  import main.js file
4.  import llama.py
5.  login to ngrok account
6.  run the node files
7.  Ngrok will give a url to make a API call to the node.js file for tunneling
8.  the user has to update the ngrok url in app.py file.
9.  the user has to install all the requirements in requirements.txt file
10. the user has to run the app.py streamlit file
    

   

#### Future scope:
      To enhance the solution’s effectiveness, we plan to incorporate litigation strategy development in the near future by fine-tuning an LLM and integrating Retrieval-Augmented Generation.
