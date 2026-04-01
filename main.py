################## Read TXT file ########################
from langchain_community.document_loaders import TextLoader 

loader=TextLoader("/home/gourav/suraj/abc.txt")
docs=loader.load()

print(loader)

print(docs)


####################### Read Pdf #############################


# from langchain_community.document_loaders import PyPDFLoader

# loader=PyPDFLoader("/home/gourav/suraj/uploads/decrypted_MedicalRecordPOC.pdf")

# docs=loader.load()

# print(docs)

from langchain_community.document_loaders import WebBaseLoader

loader=WebBaseLoader(web_paths=["https://www.python.org/"])

docs=loader.load()

print(docs)

# from langchain_community.document_loaders import WikipediaLoader

# loader=WikipediaLoader(query="Genrative AI",load_max_docs=2)
# docs=loader.load()
# print(docs)
with open("abc.txt","r") as f:
    speech=f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter =RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)

docs=text_splitter.create_documents([speech])
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)
print(docs[3].page_content)