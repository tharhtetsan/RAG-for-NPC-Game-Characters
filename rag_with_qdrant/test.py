import os
from service import VectorService


file_path = r"/Users/tharhtet/Documents/github/RAG-for-NPC-Game-Characters/rag_with_qdrant/corpus/the_wonderful_wizard_of_oz.txt"

collection_name = "knowledgebase"
collection_size = 768
chunk_size = 512 
vector_service =  VectorService()
   

vector_service.store_file_content_in_db(file_path,512,"",768)
