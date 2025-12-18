"""
LangChain + Gemini Chatbot with Advanced Memory (Modern Runnable API)
Implements: SummaryBufferMemory, EntityMemory, and VectorStore Memory
Uses LangChain Expression Language (LCEL) and Runnable interface
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*70)
print("LANGCHAIN + GEMINI CHATBOT WITH MEMORY (MODERN RUNNABLE API)")
print("="*70)
print("\nPlease enter your Google API Key:")
print("(Get it from: https://makersuite.google.com/app/apikey)")
api_key = input("API Key: ").strip()

if not api_key:
    print("\nNo API key provided. Exiting...")
    exit(1)

os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True
)

print("\n✓ Gemini model initialized successfully!")


# ============================================================================
# ASSIGNMENT 1: SUMMARY BUFFER MEMORY CHATBOT (RUNNABLE API)
# ============================================================================

class SummaryBufferChatbot:
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING: SUMMARY BUFFER MEMORY CHATBOT (RUNNABLE)")
        print("="*70)
        
        # Initialize memory
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=50,
            return_messages=True,
            memory_key="history"
        )
        
        # Create prompt using ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with conversation memory."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create runnable chain using LCEL
        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self._load_memory)
            )
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        print("✓ Summary Buffer Memory initialized with Runnable API")
    
    def _load_memory(self, inputs):
        """Load memory for the chain"""
        memory_vars = self.memory.load_memory_variables({})
        return memory_vars.get("history", [])
    
    def _save_context(self, inputs, outputs):
        """Save context to memory"""
        self.memory.save_context(
            {"input": inputs["input"]},
            {"output": outputs}
        )
    
    def chat(self, user_input):
        # Prepare inputs
        inputs = {"input": user_input}
        
        # Invoke the chain
        response = self.chain.invoke(inputs)
        
        # Save to memory
        self._save_context(inputs, response)
        
        return response
    
    def show_memory(self):
        memory_vars = self.memory.load_memory_variables({})
        print("\n--- Memory State ---")
        print(f"Buffer Length: {len(self.memory.chat_memory.messages)}")
        if hasattr(self.memory, 'moving_summary_buffer'):
            print(f"Summary: {self.memory.moving_summary_buffer}")
        return memory_vars


# ============================================================================
# ASSIGNMENT 2: ENTITY MEMORY CHATBOT (RUNNABLE API)
# ============================================================================

class EntityMemoryChatbot:
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING: ENTITY MEMORY CHATBOT (RUNNABLE)")
        print("="*70)
        
        # Initialize memory
        self.memory = ConversationEntityMemory(
            llm=llm,
            return_messages=True
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that remembers information about the user.

Known entities about the user:
{entities}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create runnable chain
        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self._load_history),
                entities=RunnableLambda(self._load_entities)
            )
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        print("✓ Entity Memory initialized with Runnable API")
    
    def _load_history(self, inputs):
        """Load conversation history"""
        memory_vars = self.memory.load_memory_variables({"input": inputs.get("input", "")})
        return memory_vars.get("history", [])
    
    def _load_entities(self, inputs):
        """Load stored entities"""
        memory_vars = self.memory.load_memory_variables({"input": inputs.get("input", "")})
        return memory_vars.get("entities", "No entities stored yet")
    
    def _save_context(self, inputs, outputs):
        """Save context to memory"""
        self.memory.save_context(
            {"input": inputs["input"]},
            {"output": outputs}
        )
    
    def chat(self, user_input):
        # Prepare inputs
        inputs = {"input": user_input}
        
        # Invoke the chain
        response = self.chain.invoke(inputs)
        
        # Save to memory
        self._save_context(inputs, response)
        
        return response
    
    def show_memory(self):
        memory_vars = self.memory.load_memory_variables({"input": ""})
        print("\n--- Stored Entities ---")
        if 'entities' in memory_vars:
            entities = memory_vars['entities']
            if entities.strip():
                print(entities)
            else:
                print("No entities stored yet")
        return memory_vars


# ============================================================================
# ASSIGNMENT 3: VECTOR STORE MEMORY CHATBOT (RUNNABLE API)
# ============================================================================

class VectorStoreChatbot:
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING: VECTOR STORE MEMORY CHATBOT (RUNNABLE)")
        print("="*70)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_texts(
            ["Conversation started"],
            self.embeddings,
            metadatas=[{"source": "initial"}]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize memory
        self.memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="history",
            input_key="input"
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to relevant past conversation context.

Relevant context from past conversations:
{history}"""),
            ("human", "{input}")
        ])
        
        # Create runnable chain
        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self._load_memory)
            )
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        print("✓ Vector Store Memory initialized with Runnable API")
    
    def _load_memory(self, inputs):
        """Load relevant memories from vector store"""
        memory_vars = self.memory.load_memory_variables({"input": inputs.get("input", "")})
        return memory_vars.get("history", "")
    
    def _save_context(self, inputs, outputs):
        """Save context to memory"""
        self.memory.save_context(
            {"input": inputs["input"]},
            {"output": outputs}
        )
    
    def chat(self, user_input):
        # Prepare inputs
        inputs = {"input": user_input}
        
        # Invoke the chain
        response = self.chain.invoke(inputs)
        
        # Save to memory
        self._save_context(inputs, response)
        
        return response
    
    def show_memory(self):
        print("\n--- Vector Store State ---")
        try:
            memory_vars = self.memory.load_memory_variables({"input": "test query"})
            print(f"Stored vectors: {len(self.vectorstore.docstore._dict)}")
            print(f"Retrieved context: {memory_vars.get('history', 'None')[:200]}...")
        except Exception as e:
            print(f"Error retrieving memory: {e}")


# ============================================================================
# STREAMING SUPPORT (BONUS)
# ============================================================================

class StreamingSummaryBufferChatbot(SummaryBufferChatbot):
    """Enhanced version with streaming support"""
    
    def chat_stream(self, user_input):
        """Stream the response token by token"""
        inputs = {"input": user_input}
        
        print("\n🤖 Bot: ", end="", flush=True)
        full_response = ""
        
        for chunk in self.chain.stream(inputs):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()  # New line after streaming
        
        # Save to memory
        self._save_context(inputs, full_response)
        
        return full_response


# ============================================================================
# INTERACTIVE CHAT INTERFACE
# ============================================================================

def interactive_chat(chatbot_type, use_streaming=False):
    """Interactive command-line chat interface"""
    
    # Initialize the appropriate chatbot
    if chatbot_type == '1':
        if use_streaming:
            chatbot = StreamingSummaryBufferChatbot()
            bot_name = "Summary Buffer Memory Bot (Streaming)"
        else:
            chatbot = SummaryBufferChatbot()
            bot_name = "Summary Buffer Memory Bot (Runnable)"
    elif chatbot_type == '2':
        chatbot = EntityMemoryChatbot()
        bot_name = "Entity Memory Bot (Runnable)"
    elif chatbot_type == '3':
        chatbot = VectorStoreChatbot()
        bot_name = "Vector Store Memory Bot (Runnable)"
    else:
        print("Invalid choice!")
        return
    
    print(f"\n{'='*70}")
    print(f"CHATTING WITH: {bot_name}")
    print("="*70)
    print("\nCommands:")
    print("  - Type your message and press Enter to chat")
    print("  - Type 'memory' to view current memory state")
    print("  - Type 'quit' or 'exit' to end conversation")
    print("="*70 + "\n")
    
    message_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            if user_input.lower() == 'memory':
                chatbot.show_memory()
                continue
            
            if not user_input:
                continue
            
            # Get bot response
            if use_streaming and hasattr(chatbot, 'chat_stream'):
                response = chatbot.chat_stream(user_input)
            else:
                print("\n🤖 Bot: ", end="", flush=True)
                response = chatbot.chat(user_input)
                print(response)
            
            message_count += 1
            
            # Show memory hint every 5 messages
            if message_count % 5 == 0:
                print("\n💡 Tip: Type 'memory' to see what I remember!")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


# ============================================================================
# DEMO MODE - RUN ALL ASSIGNMENTS
# ============================================================================

def run_demo():
    """Run automated demo of all three memory types"""
    
    print("\n" + "="*70)
    print("DEMO MODE: Testing All Memory Types (Runnable API)")
    print("="*70)
    
    # Demo 1: Summary Buffer
    print("\n\n[DEMO 1/3] Summary Buffer Memory (Runnable)")
    print("-" * 70)
    bot1 = SummaryBufferChatbot()
    
    test_inputs_1 = [
        "Hi! My name is Alex and I'm a software engineer.",
        "I love programming in Python and building AI applications.",
        "I also enjoy hiking on weekends.",
        "What do you know about me?"
    ]
    
    for inp in test_inputs_1:
        print(f"\nYou: {inp}")
        response = bot1.chat(inp)
        print(f"Bot: {response}")
    
    bot1.show_memory()
    
    # Demo 2: Entity Memory
    print("\n\n[DEMO 2/3] Entity Memory (Runnable)")
    print("-" * 70)
    bot2 = EntityMemoryChatbot()
    
    test_inputs_2 = [
        "My name is Sarah and I'm 28 years old.",
        "I work as a data scientist at Google.",
        "My favorite color is blue and I love Italian food.",
        "What are my preferences?"
    ]
    
    for inp in test_inputs_2:
        print(f"\nYou: {inp}")
        response = bot2.chat(inp)
        print(f"Bot: {response}")
    
    bot2.show_memory()
    
    # Demo 3: Vector Store
    print("\n\n[DEMO 3/3] Vector Store Memory (Runnable)")
    print("-" * 70)
    bot3 = VectorStoreChatbot()
    
    test_inputs_3 = [
        "I'm planning a trip to Japan next spring.",
        "I'm interested in visiting temples and trying authentic sushi.",
        "I also want to see Mount Fuji.",
        "What should I do on my vacation based on what I told you?"
    ]
    
    for inp in test_inputs_3:
        print(f"\nYou: {inp}")
        response = bot3.chat(inp)
        print(f"Bot: {response}")
    
    bot3.show_memory()
    
    print("\n" + "="*70)
    print("✓ ALL DEMOS COMPLETED!")
    print("="*70)


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main application entry point"""
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU (MODERN RUNNABLE API)")
        print("="*70)
        print("\nChoose an option:")
        print("  1. Chat with Summary Buffer Memory Bot")
        print("  2. Chat with Entity Memory Bot (User Preference Tracking)")
        print("  3. Chat with Vector Store Memory Bot")
        print("  4. Chat with Streaming Support (Summary Buffer)")
        print("  5. Run Automated Demo (All Three Bots)")
        print("  6. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '6':
            print("\n👋 Thank you for using the chatbot! Goodbye!")
            break
        elif choice == '5':
            run_demo()
        elif choice == '4':
            interactive_chat('1', use_streaming=True)
        elif choice in ['1', '2', '3']:
            interactive_chat(choice)
        else:
            print("\n❌ Invalid choice! Please enter 1-6.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Valid Google API key")
        print("  3. Active internet connection")
        print("\n📚 LangChain Runnable API Documentation:")
        print("  https://python.langchain.com/docs/expression_language/")