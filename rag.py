import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from openai import OpenAI



class faiss_rag :
    def __init__(self,faiss_index_path) -> None:
        self.loaded_retriever = self.loadEmbed(faiss_index_path=faiss_index_path)
            
    
    # Load embeddings for RAG
    def loadEmbed(self,faiss_index_path : str):
        """
        The embeddings are retrived from a local storage.
        Will be good if can access the embedding from S3 bucket.
        """
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        db= FAISS.load_local(faiss_index_path,embeddings=embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        return retriever
    

    # Create a summarizer for the similary searching outout
    def summarizeRag(self):
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
        summarizer =pipeline(
            "summarization",
            model=model_name,
            tokenizer=tokenizer,
            return_tensors='pt'
        )
        return tokenizer, summarizer
    

    # Combine RAG similary search and summarizer
    def prep4Prompt(self,question:str):
        docs = self.loaded_retriever.get_relevant_documents(question)
        line = docs[0].page_content
        tokenizer, summarizer = self.summarizeRag()
        response = summarizer(line, max_length=200, min_length=50, do_sample=False)
        token_ids = response[0]['summary_token_ids'].numpy()
        decoded_text = tokenizer.decode(token_ids)
        cleaned_text = re.sub(r'[^\w\s.,;\'"\-?!]', '', decoded_text)
        return cleaned_text

    
    # Get conversation output
    def chatbotQA(self,client,universe, question, character_asking, character_answering, sentiment):
        completion = client.chat.completions.create(
        model= "gpt-3.5-turbo", # this field is currently unused
        messages=self.promptEngineerQA(universe,question, character_asking, character_answering, sentiment),
        temperature=0.8,
        )
        return completion.choices[0].message.content
    

    def get_completion_usingLLMStudio(self,universe, question, character_asking, character_answering, sentiment):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        messages=self.promptEngineerQA(universe,question, character_asking, character_answering, sentiment)
        completion = client.chat.completions.create(
            model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            messages=messages,
            temperature=0.7,
            )
        
        return completion.choices[0].message.content

    
    # Create prompt with engineering for Q&A
    def promptEngineerQA(self,universe,question, character_asking, character_answering, sentiment):
        """
        The prompt engineering is still under developments. Will need modification later.
        """
        cleaned_rag = self.prep4Prompt(question)
        print("cleaned_rag : ",cleaned_rag)
        print()

        prompt_template_nokeyword = [
            {"role": "system", "content": \
            f"Act as if you were {character_answering}, from {universe}. \
            Next, you will be given some context to a question {character_asking} will ask. \
            Your emotions towards {character_asking} are {sentiment}. \
            Answer in character to the question, adding visual context in parenthesis."},
            {"role": "user", "content": f"<context>{cleaned_rag} </context> <question>{question} </question>"}
        ]

        print("prompt_template_nokeyword : ",prompt_template_nokeyword)

        return prompt_template_nokeyword
    
    
    