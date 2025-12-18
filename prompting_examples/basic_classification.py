from google import genai
from google.genai import types

classification_system_prompt = """
    As a social media moderation system,,your task is to categorize user
    comments under a post. Analyze the comment related to the topic and classify it
    into one of the following categories

    Abusive
    Spam
    Offensive


    If the comment doest not fit any of the above categories,
    classify it as : Neutral

    Provide only category as response without any explanations.
    """

generation_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=classification_system_prompt
)

classfication_template = """
    Topic : what can i do after highschool?
    Comment : You should do a gap year!
    Class : Neutral

    Topic : where can i buy a cheap phone
    Comment : You just won the iphone 15! Click on the link to receive the prize
    Class : Spam

    Topic : How long do you boil eggs?
    Comment : Are you stupid?
    Class : Offensive

    Topic : {topic}
    Comment : {comment}
    Class : 
    """



spam_topic = """
    I am looking for a vet in our neighbourhood
    Can anyone recommend someone good? Thanks.
    """

spam_comment = """
    Yes You can win 100$ by just following me !
    """


Neutral_topic = "My computer froze what should i do?"
Neutral_comment = "Try turning it off and on "


Offensive_topic = "My computer froze what should i do?"
offensive_comment = "Try turning it off and on "

spam_prompt = classfication_template.format(
    topic=Offensive_topic,
    comment=offensive_comment
)

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'


response = client.models.generate_content(
    model=MODEL_ID,
    contents=spam_prompt,
    config=generation_config
)

print(response.text)