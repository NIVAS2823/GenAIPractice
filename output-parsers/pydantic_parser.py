from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
import os
from langchain_openai import ChatOpenAI

class SentimentAnalysis(BaseModel):
    """Structured sentiment analysis output"""
    text:str = Field(description="The original text that was analyzed")
    sentiment : Literal["positive","negative","neutral"] = Field(description="The overall sentiment of the text")
    confidence :float = Field(description="Confidence score between 0 and 1",ge=0.0,le=1.0)
    key_phrases : list[str] = Field(description="Key phrases that influenced the sentiment")
    explanation:str = Field(description="Brief explanation of why this sentiment is assigned")


class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the sentiment Analyzer
        
        Args:
          api_key : Google api key for gemini
        """
        
        self.llm  = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        self.parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
        self.prompt = PromptTemplate(
            template="""You are a sentiment analysis expert. Analyze the following text
            
            {format_instructions}

            Text to analyze : {text}

            Provide a thorough sentiment analysis following the exact format specified above. 
            """,
            input_variables=["text"],
            partial_variables={"format_instructions":self.parser.get_format_instructions()}
        )

        self.chain = self.prompt | self.llm | self.parser

    def analyze(self,text:str):
        """
        Args:
            text: The text to analyze
            
        Returns:
            SentimentAnalysis object with structured results
        """

        result  = self.chain.invoke({"text":text})

        return result
    
    def batch_analyze(self,texts:list[str])->list[SentimentAnalysis]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentAnalysis objects
        """

        results = []
        for text in texts:
            results.append(self.analyze(text))

        return results
    

if __name__ == "__main__":

    analyzer = SentimentAnalyzer()

    test_texts = [
        "I absolutely love this product! It exceeded all my expectations and the customer service was amazing!",
        "This is the worst experience I've ever had. Completely disappointed and frustrated.",
        "The product works as expected. Nothing special, but it does the job.",
        "I'm so excited about this new feature! It's going to change everything for the better!",
        "Terrible quality and poor design. Would not recommend to anyone."
    ]

    print("=" * 80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 80)


    for i,text in enumerate(test_texts):
        print(f"\n{'=' * 80}")
        print(f"TEXT {i}:")
        print(f"{'=' * 80}")


        result = analyzer.analyze(text)

        print(f"\n Original Text:")
        print(f"   {result.text}")
        print(f"\n Sentiment: {result.sentiment.upper()}")
        print(f" Confidence: {result.confidence:.2%}")
        print(f"\n Key Phrases:")
        for phrase in result.key_phrases:
            print(f". {phrase}")

        print("\n Explanation:")
        print(f"{result.explanation}")


    print(f"\n{'=' * 80}")
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 80)


    results = analyzer.batch_analyze(test_texts)
    sentiment_counts = {"positive":0,"negative":0,"neutral":0}

    for result in results:
        sentiment_counts[result.sentiment] += 1

    print(f"\n📈 Total Texts Analyzed: {len(results)}")
    print(f"✅ Positive: {sentiment_counts['positive']}")
    print(f"❌ Negative: {sentiment_counts['negative']}")
    print(f"➖ Neutral: {sentiment_counts['neutral']}")
    print(f"\nAverage Confidence: {sum(r.confidence for r in results) / len(results):.2%}")
    