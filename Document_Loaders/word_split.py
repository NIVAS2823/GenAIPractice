from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

word_file = "Nivas_Mididhodi_Resume.docx"
output_file = "word_doc_summary.txt"

loader = UnstructuredWordDocumentLoader(word_file)

docs = loader.load()

text = "\n".join([doc.page_content for doc in docs])


splitter = RecursiveCharacterTextSplitter(
    chunk_size =300,
    chunk_overlap = 60
)

chunks = splitter.split_text(text)

# print(len(chunks))
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this chunk :\n {text}\n"
)


chain = prompt | llm

with open(output_file,"w",encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        result = chain.invoke({"text":chunk}).content
        f.write(f"\n Chunk {i+1}---\n")
        f.write(f"Original : \n {chunk}\n")
        f.write(f"Summary : \n {result}\n")
        
print(f"The Dcocument {word_file} summary is saved to {output_file}")
